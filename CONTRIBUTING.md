# Contributing

How development works in dfo-ai. Read this before opening a branch.

## Getting set up

You need [`just`](https://github.com/casey/just), [`uv`](https://docs.astral.sh/uv/),
and the `gcloud` CLI, plus access to the `dev-fin-observatory` GCP project.

```sh
gcloud auth login
gcloud auth application-default login
just setup    # env file, dependencies, pre-commit hooks
just dev      # harness on :8080
```

Recipes that call dfo-apis (`just dev`, `just studio-live`,
`just eval-planner-entry-record`, `just graph-schema`) reach it at
`DFO_APIS_URL`, `http://localhost:8083` by default. Cloud Run IAM guards the
dev deployment, so serve it there from a second terminal:

```sh
just apis-proxy   # gcloud run services proxy dfo-dev-apis --region northamerica-northeast1 --project dev-fin-observatory --port 8083
```

The proxy signs each request as your gcloud account, which needs
`roles/run.invoker` on `dfo-dev-apis`. A dfo-apis checkout's `just dev` serves
the same port instead.

`just` on its own lists every recipe. `just doctor` checks the whole local
setup -- tooling, gcloud auth, which config keys resolved, whether the
workspace is synced -- and tells you the command to fix anything it finds.

`just setup` is safe to re-run. It skips what is already done.

## Configuration

Local config lives in `.env` at the workspace root. It is generated and
gitignored; `just env` builds it from `.env.example`.

Two kinds of value live there.

**Secrets** come from Secret Manager in `dev-fin-observatory`:

| Key                                          | Secret                                                          |
| -------------------------------------------- | --------------------------------------------------------------- |
| `GOOGLE_API_KEY`                             | `dfo-dev-gemini-api-key` (shared with `dfo-dcp` and `dfo-apis`) |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | `dfo-dev-langfuse-public-key`, `dfo-dev-langfuse-secret-key`    |

**Everything else** (`MODEL_NAME`, `DFO_APIS_URL`, `LANGFUSE_BASE_URL`, `PORT`)
is a committed default in `.env.example`, so git distributes it.
`just env` carries new keys into an existing `.env` on the next run, and
`just doctor` reports when yours has fallen behind.

Each key resolves in order: a value already in `.env`, then an exported shell
variable, then Secret Manager. Existing values survive, so re-running
`just env` after an edit is safe, and a one-off override needs no file edit:

```sh
MODEL_NAME=gemini-3-pro just dev
```

With any Langfuse credential missing, the run starts anyway. Traces go to
`OTEL_EXPORTER_OTLP_ENDPOINT` when it is set, and tracing is off otherwise.
`/healthz` reports the resolved state as `telemetry: langfuse`, `otel` or `off`.

To rotate a credential, add a new version to the secret in Secret Manager,
delete that line from your `.env`, and run `just env`.

### Adding a config key

1. Add it to `.env.example`. Give it a real default if it isn't a secret.
1. Add the field to `BaseServiceSettings` (`packages/common`) if every service
   reads it, or to that service's own `Settings` subclass if one does.
1. For a secret, create it in Secret Manager as `dfo-dev-<name>` and add a
   `resolve` line to the `env` recipe in the `justfile`.

## Day to day

```sh
just check          # lint, typecheck, test -- CI's lint and test jobs
just fix            # apply ruff lint fixes and formatting
just test           # the whole suite from the workspace root
just docker-build   # the Cloud Run image, the way CI's docker job builds it
```

`just check` covers CI's `lint` and `test` jobs, so local green predicts both
going green. CI runs a third job, `docker`, which builds the Cloud Run image.
`just check` leaves it out because an image build is too slow for the common
path. Reach for `just docker-build` when you touch `deploy/Dockerfile.ai` or a
dependency that lands in the image. There is also a `test-<member>` shortcut
per workspace member (`just test-harness`, `just test-data-mcp`, and so on).

Plain `uv sync` only syncs the workspace root, which is virtual and publishes
nothing. Use `just sync` to install every member into the shared `.venv`.

## Branch model

`dev` is the trunk. Branch off it, and open pull requests into it. CI
(`lint`, `test`, `docker`) runs on every pull request into `dev` or `main`
once it is marked ready, and again on each push to `dev`. A draft pull request
runs no CI.

Merging to `dev` deploys to the dev Cloud Run service.

A release is a pull request from `dev` into `main`. `deploy-prod.yml` deploys
production on every push to `main`.

## Conventions

Tests run from an empty working directory, so a local `.env` never becomes an
input to the suite (`conftest.py`).

`data-mcp`'s digest corpus (`services/data-mcp/tests/digest_corpus.py`) measures the
model-facing character cost of every semantic status the search door can render,
against a committed baseline and a per-status budget. The budgets only fail on real
inflation. A deliberate digest change that moves the recorded numbers means running
`just digest-baseline` so the committed record stays current -- the regenerated diff
to `digest_baseline.json` is what reviewers read.

Settings are `pydantic-settings` classes, one per service, all subclassing
`common.settings.BaseServiceSettings`. Read config through them, not `os.getenv`
directly. The exception is anything a third-party library reads from the
environment itself, such as `GOOGLE_API_KEY`, which google-adk picks up.

`data-mcp`'s definition store persists the definitions `search` mints
until the first `get_data` registers them permanently. `DEFINITION_STORE_BUCKET`
unset backs it with an in-memory store instead of GCS -- fine for local dev
and for a single Cloud Run instance, but every other instance behind the
same service then refuses `get_data` with `definition_not_held` for an id
this instance minted, so `server.py`'s lifespan logs a warning when the
bucket is unset and `K_SERVICE` (Cloud Run's own service marker) is set.

`test_scripts/` is for one-off local checks: probes against a live service,
demo drivers, ad hoc scripts you want to keep for a while. It is ignored by
git and is not part of the test suite; the suite is `services/*/tests` and
`packages/*/tests`, and only those run under `just check` and in CI.

Ruff and `ty` settings live in the root `pyproject.toml` and apply to every
member. Pre-commit runs both, plus `just typecheck`.

### Upgrading `google-adk`

The harness pins `google-adk==2.9.1` (`services/harness/pyproject.toml`) and depends on four
of its private modules: `flows/llm_flows/contents` (`_is_event_belongs_to_branch`),
`flows/llm_flows/_fencing` (`_present_other_agent_message`), `tools/agent_tool`
(`_SingleTurnAgentTool`) and `workflow/_llm_agent_wrapper` (`prepare_llm_agent_context`).
Nothing imports them, so an upgrade that changes their behaviour produces no import error.
The planner's context isolation rests on all four: its run is stamped with a branch, and
ADK's branch filter drops the translator's events from what the planner is handed. The
branch comparison direction, the sub-branch naming and the `single_turn` scope inheritance
are each one of those modules' contract.

ADK's filter is fail-open. `_is_event_belongs_to_branch` opens with
`if not invocation_branch or not event.branch: return True`, so it keeps an event whose
branch is empty. The planner is isolated while `override_branch=_PLANNER_BRANCH` reaches
`ctx.run_node`, which happens at one call site, in
`services/harness/src/harness/planner_seam.py`. After each run, and on the way out of one
that suspends for a planner-side ask, `harness.planner_seam._check_branches` checks that every
planner event carries `_PLANNER_BRANCH` and every translator event a branch strictly under it,
and that a completed run appended a planner event at all. It raises `PlannerSeamError`
otherwise. A dropped keyword, a second call site without it, or an ADK release that stops
honouring it therefore fails the run. The relay ceiling below is the measured consequence.

`services/harness/tests/test_context_budget.py` is the first test file to fail after a bump.
`just check` runs it with the rest of the suite. Two of its tests are the early warning on a
bump: `test_no_relayed_foreign_agent_text_reaches_the_planner` holds the fenced
foreign-agent relay in the planner's context at exactly zero chars, and
`test_the_relay_check_measures_a_relay_that_is_there` proves that reading can go non-zero,
so the zero is a measurement, not a bucket that stopped reading.
`test_the_cached_prefix_is_stable_across_a_run` holds the planner's static prefix
byte-identical across a run, which is what Gemini's implicit caching matches across
requests and what the shared explicit cache keys on (`harness.prefix_cache.prefix_key`).

A failure here after a bump means ADK changed one of those behaviours. The failing test's
own docstring names the mechanism it asserts, and `.venv/lib/python3.*/site-packages/google/adk`
is the installed source to check that claim against. An accommodation to the new behaviour
that moves a measured number means running `just context-baseline`, and the diff of
`services/harness/tests/context_baseline.json` is what reviewers read.

## Deploying

CI builds one image and updates the dev Cloud Run service on every merge to
`dev`, and production on every push to `main`. `infra` owns the service
definition, including its runtime secrets. The `.env` flow above is for local
development. See [deploy/README.md](./deploy/README.md) for
the full division and the GitHub Environment setup.
