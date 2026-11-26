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

Recipes that call `apis` (`just dev`, `just studio-live`, `just eval-planner-entry-record`, `just graph-schema`) reach it at `DFO_APIS_URL`, `http://localhost:8083` by default. Cloud Run IAM guards the dev deployment, so serve it on that port from a second terminal:

```bash
just apis-proxy   # gcloud run services proxy dfo-dev-apis --region northamerica-northeast1 --project dev-fin-observatory --port 8083
```

The proxy signs each request as your gcloud account, which needs `roles/run.invoker` on `dfo-dev-apis`. A local `apis` checkout's `just dev` serves the same port instead.

`just setup` is safe to re-run and skips what is already done. It checks prerequisites, builds `.env`, syncs the workspace members into the shared `.venv`, and installs pre-commit hooks. `just doctor` checks tooling, gcloud auth, which config keys resolved, and whether the workspace is synced, and prints the command that fixes anything it finds.

## Configuration

Local config lives in `.env` at the workspace root. It is generated and gitignored, and `just env` builds it from `.env.example`. Two kinds of value live there.

Secrets come from Secret Manager in `dev-fin-observatory`:

| Key                                          | Secret                                                       |
| -------------------------------------------- | ------------------------------------------------------------ |
| `GOOGLE_API_KEY`                             | `dfo-dev-gemini-api-key` (shared with `dcp` and `apis`)      |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | `dfo-dev-langfuse-public-key`, `dfo-dev-langfuse-secret-key` |

Everything else (`MODEL_NAME`, `DFO_APIS_URL`, `LANGFUSE_BASE_URL`, `PORT`) is a committed default in `.env.example`, so git distributes it. `just env` carries new keys into an existing `.env` on the next run, and `just doctor` reports when yours has fallen behind.

Each key resolves in order: a value already in `.env`, then an exported shell variable, then Secret Manager. Existing values survive, so re-running `just env` after an edit is safe, and a one-off override needs no file edit. Set `MODEL_NAME` and `PLANNER_MODEL_NAME` together, since the harness refuses two models on different surfaces:

```bash
MODEL_NAME=gemini-3.8-flash PLANNER_MODEL_NAME=gemini-3.8-flash just dev
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

## Branches and deploys

`dev` is the trunk. CI (`lint`, `test`, `docker`) runs on every ready pull request into `dev` or `main`, and again on each push to `dev`. Merging to `dev` deploys the dev Cloud Run service. A release is a pull request from `dev` into `main`, and `deploy-prod.yml` deploys production on every push to `main`.

CI builds one image and updates the existing Cloud Run service. The `infra` repository owns the service definition, including its runtime secrets. The `.env` flow above is for local development only. [deploy/README.md](../deploy/README.md) has the full division of responsibilities and the GitHub Environment setup.
