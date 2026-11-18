# Deploy

GitHub Actions in this repo builds the `ai` image (the harness plus the
data-mcp, analysis-mcp, and presentation-mcp FastMCP apps it mounts
in-process) and swaps it
onto an existing Cloud Run service. It never creates or reconfigures a
service. Dev deploys automatically on every push to `dev` that touches the
image's inputs. Production deploys on every push to `main`, and on manual
dispatch. Both run the same steps from `build-deploy.yml`. The sibling
`infra` repo owns the Cloud Run service definition (service account,
scaling, secrets, env vars) that these workflows target.

The workspace services (harness, data-mcp, analysis-mcp,
presentation-mcp) deploy as ONE Cloud Run service. See docs/architecture.md for the full rationale.

Until infra defines the Cloud Run service and someone sets the repo
variables below, the deploy workflows will fail at the "resolve image ref"
or `auth` step. `ci.yml` needs none of this and works today.

## Image naming

```
${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/dfo-ai:<tag>
```

`<tag>` is the git SHA the image was built from. Dev builds one image per
commit to `dev` that touched packages/, services/, deploy/Dockerfile.ai, or
either deploy workflow. Production builds one per push to `main`, into
prod's own registry.

## GitHub vars and secrets

`build-deploy.yml` reads these from a GitHub Environment matching the
deploy target (`dev`, later `staging`, `production`). Configure the same variable
and secret _names_ in each environment with per-environment values. The
workflow YAML doesn't change between environments. Only the environment it
targets does.

| Name                  | Kind   | Example value                                                                               | Who provides it                                                   |
| --------------------- | ------ | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `WIF_PROVIDER`        | secret | `projects/123456789/locations/global/workloadIdentityPools/github/providers/github-actions` | the infra repository's handoff (WIF pool/provider)                |
| `WIF_SA`              | secret | `dfo-ai-deployer@dev-fin-observatory.iam.gserviceaccount.com`                               | the infra repository's handoff (deployer service account)         |
| `REGION`              | var    | `northamerica-northeast1` (dev), `europe-west1` (production)                                | the infra repository's handoff                                    |
| `PROJECT_ID`          | var    | `dev-fin-observatory` (dev); the prod project ID in production                              | the infra repository's handoff                                    |
| `AR_REPO`             | var    | `datacommons`                                                                               | the infra repository's handoff (Artifact Registry repo)           |
| `SERVICE_AI`          | var    | `dfo-dev-ai`                                                                                | the infra repository's handoff, once the Cloud Run service exists |
| `JOB_LANGFUSE_EXPORT` | var    | `dfo-prod-langfuse-export` (production only)                                                | the infra repository's handoff, once the job exists               |

The Cloud Run service itself also needs `STUDIO_API_TOKEN` mounted as a
secret (infra, in the service definition next to `GOOGLE_API_KEY`). It
is the shared secret the frontend's proxy presents on studio requests. A
service without it accepts any non-empty bearer token and logs
`STUDIO AUTH IS OPEN` at startup.

The service takes `LIVE_REFRESH_PRINCIPAL` as a second secret (infra
generates it in the apis stack and mounts one value on apis, this service,
and the website). It is the bearer of `POST /api/studio/refresh`, which the
website calls to redraw a live share's derived card, and the principal this
service presents to apis' frame refresh and render routes. presentation-mcp,
mounted in the same process, reads the same variable to rebuild a lost summed
or cut frame for a public `/mcp` client. A service
without it refuses every refresh with 401 and logs `LIVE REFRESH IS CLOSED` at
startup, and a lost summed or cut frame refuses `rendering_not_recorded`. Startup fails when it equals `STUDIO_API_TOKEN`. The route also reads
`DFO_APIS_URL` and `DATASETS_FIRST_PARTY_PRINCIPAL`, which the service already
holds.

`DATASETS_FIRST_PARTY_PRINCIPAL` is the principal apis grants publish to, and
must equal apis' own value. presentation-mcp sends it when it publishes a
derived frame, and on the card creates and revises of Studio conversations,
which reach it over the harness's loopback mount. apis exempts a principal
holding the publish grant from the per-caller card allowance, so Studio's
cards spend none. A card an external MCP client writes through `/mcp` never
carries it.

`DFO_CALLER_PRINCIPAL` is the secret apis shares with the website, and must
equal apis' and the website's value. The website names each `/mcp` client
in `x-dfo-caller` as `ip:<digest>`. analysis-mcp sends the secret and that
caller with each analysis runner call, and presentation-mcp with each card an
external client creates or revises, so apis counts the client's own
allowance. Unset, neither sends a caller: runner calls are not attributed to
a client, and every external client's cards spend the one allowance of the
address apis sees for this service.

A refresh answers within 53 seconds, the deadline `LIVE_REFRESH_DEADLINE_S`
(default 45) plus an 8 second hard stop. The website gives the call 60 seconds, so startup
fails for a deadline above 52 or below 30, and the Cloud Run request timeout,
set in infra, must be above the deadline plus 8. The other
refresh settings (`LIVE_REFRESH_REQUESTS_PER_S`, `LIVE_REFRESH_BURST`,
`LIVE_REFRESH_MAX_IN_FLIGHT`, `LIVE_REFRESH_ROWS_IN_FLIGHT_MAX`) have defaults
sized for one instance and are described in `services/harness/README.md`.

`MCP_API_TOKEN` needs no secret on this one-Cloud-Run-service deployment.
Left unset, the harness mints a token at startup and hands it to its own
`data-mcp` / `analysis-mcp` / `presentation-mcp` MCP clients (`common.auth`),
so the three mounts require a credential nothing outside the process ever
holds. Set it as a secret only once one of those services is split out to
its own Cloud Run service (see "Splitting a service out later" below).
`DATA_MCP_URL` / `ANALYSIS_MCP_URL` / `PRESENTATION_MCP_URL` then point off
loopback, and startup refuses that combination without a shared
`MCP_API_TOKEN` the split-out service also has.

The workspace's `dfo-spec-kit` git dependencies come from the public
`DevelopmentFinanceObservatory/spec-kit` repo, so `ci.yml`, `build-deploy.yml`,
and `deploy/Dockerfile.ai` need no token to install them.

The service name value above is a placeholder for illustration. infra
names the actual service, and whatever they pick goes in this var verbatim.

infra sets every variable and secret in the Environment: `SERVICE_AI`,
`JOB_LANGFUSE_EXPORT` (production), `WIF_PROVIDER`, `WIF_SA`, `REGION`, `PROJECT_ID`, and `AR_REPO`. Run
`just ENV=dfo-dev handoff ai` from the infra repo for the `dev`
Environment, and `just ENV=dfo-prod handoff ai` for `production`. The
script also sets the deployment branch policy. The values stay reproducible
and the Environment has one writer.

## GCP prerequisites (infra)

Before the first real deploy, infra needs to set up, per environment:

- A Workload Identity Federation pool and provider trusting this repo's
  GitHub Actions OIDC issuer (no JSON service account keys).
- A deployer service account, impersonable through that provider, with:
  - `roles/run.developer` (update Cloud Run services)
  - `roles/artifactregistry.writer` (push images)
  - `roles/iam.serviceAccountUser` (act as the Cloud Run runtime SA, which is
    a dedicated identity on the `ai` service)
  - in production, `roles/run.developer` on the Langfuse export job, to
    point it at each new image, and `roles/iam.serviceAccountUser` on the
    job's service account
- ONE Cloud Run service for the `ai` image (the harness process that
  mounts data-mcp, analysis-mcp, and presentation-mcp in-process, per
  docs/architecture.md),
  including its runtime service account, scaling config, env vars, and
  secrets. This repo's workflows only ever update the `--image` on the
  service that already exists.
- Model credentials on that service. The harness uses the Gemini developer API
  key, `GOOGLE_API_KEY`, which google-adk reads from the process environment, so
  wire it into the service definition as a secret. The harness refuses to build
  the planner when its Gemini model would call Vertex AI, which rejects the
  planner's tool declarations. google-genai decides that from
  `GOOGLE_GENAI_USE_ENTERPRISE` when it is set (`true` or `1`, in any case), and
  otherwise from the deprecated `GOOGLE_GENAI_USE_VERTEXAI`, read the same way.
- `DFO_APIS_URL` on that service, pointing at apis, the FastAPI app that
  serves both the Search and Datasets surfaces. The data-mcp component
  mounted in the harness reads it to reach that API and falls back to a
  `http://localhost:8083` default without it, so the deployed service can't
  search until it's set. The value depends on the apis dev deployment
  existing. The Datasets surface additionally needs the apis service's
  `DFO_DATASETS_SURFACE` flag set, since that gates whether it mounts at all.
- `DFO_SITE_ORIGIN` on that service, set to the website's public origin
  (scheme and host, no path). The artifact download links data-mcp hands to
  models sit on it, since the website proxies `/api/datasets/*` to dfo-apis
  with its own ID token and dfo-apis refuses callers without one. The harness
  fails at startup, naming the variable, when it is unset on Cloud Run.
- `PRESENTATION_PUBLIC_URL` on that service, set to the service's own public
  `https` URL plus `/mcp/presentation`. Every presentation view's payload
  names its theme stylesheet under it, and the renderer's CSP admits its
  origin. It has no default. The harness fails at startup, naming the
  variable, when it is unset or not a valid absolute `http(s)` URL with no
  query, fragment, or credentials, so a revision without it never serves
  views a browser would render unstyled.
- The Artifact Registry repo named by `AR_REPO` in each environment's
  project (`datacommons` in dev).
- In production, the Cloud Run job named by `JOB_LANGFUSE_EXPORT` on the
  `ai` image, triggered hourly with an 1800-second task timeout and no task
  retries, running as its own service account with `roles/bigquery.jobUser`
  on the project and `roles/bigquery.dataEditor` on the export dataset, and
  with the settings listed under "Langfuse export".

## Repo-side prerequisites

- A `dev` GitHub Environment with `dev` as its only deployment branch, and a
  `production` GitHub Environment with `main` as its only deployment branch
  and `release` as required reviewers.
- The vars/secrets table above, populated per environment.
- Staging is unbuilt. infra has no staging Terraform tree or GCP
  project, and standing one up means a new `infra/<env>/` directory with
  its own backend and state prefix. Once it exists, a `staging` GitHub
  Environment and a `staging` entry in `deploy-dev.yml`'s `deploy` job
  `environment` list are the only changes needed on this side.

## Splitting a service out later

The one-Cloud-Run-service model is a starting point, not a permanent
constraint. Each workspace package (harness, data_mcp, analysis_mcp,
presentation_mcp) stays a separate uv package with its own pyproject.toml, so pulling one back
out into its own Cloud Run service is additive, not a rewrite:

1. Add a `deploy/Dockerfile.<service>` for it, following
   `deploy/Dockerfile.ai`'s two-stage uv pattern: a builder stage that
   copies in the workspace manifest and the member `pyproject.toml` files
   the split-out service actually needs, runs `uv sync --frozen --no-dev --no-install-workspace --package <service>` for the dependency-only
   layer, then copies in `src/` and runs `uv sync --frozen --no-dev --no-editable --package <service>` for the full build; and a slim
   runtime stage that copies only `/app/.venv` from the builder.
1. Add a build and deploy step for it in `build-deploy.yml` pointing at its
   own `vars.SERVICE_<NAME>`.
1. Point the harness's `DATA_MCP_URL` / `ANALYSIS_MCP_URL` /
   `PRESENTATION_MCP_URL` env var at the new service's `https` URL instead
   of the in-container loopback path (a Cloud Run service URL already is
   one). `harness.settings.Settings` reads that env override in place of its
   loopback default, so no code change is needed on the harness side.
1. Provision `MCP_API_TOKEN` as a shared secret on both the harness and the
   split-out service. The harness's own minted-token default only works
   while every mount shares its process. Once one is reachable over a real
   URL, `harness.settings.Settings` refuses to start without an explicit
   `MCP_API_TOKEN` (the split-out service needs the identical value), and
   refuses a non-`https` URL even with one set -- that shared bearer goes
   out on every request, so cleartext HTTP would leak it.

The triggers for actually doing this: external MCP traffic (through the
website proxy) interfering with chat-session latency on the shared
instance, or the search/analysis and harness workloads needing to scale
independently of each other.

## Recipe backfill

`python -m harness.recipe_backfill` stores in apis the recipe of every derived frame in the materialisation bucket. A derived frame published without its recipe can only be replayed from the traces in that bucket, so run this job to completion before the bucket's 30-day age rule is restored.

The module ships in the image (`services/harness/src/harness/recipe_backfill.py`) and starts no service, no model, and no MCP server. It reads three settings from the environment or `.env`, under the names the services use, and exits 2 naming any that is unset:

| Setting                          | Use                                              |
| -------------------------------- | ------------------------------------------------ |
| `DFO_APIS_URL`                   | apis' origin                                     |
| `DATASETS_FIRST_PARTY_PRINCIPAL` | the publish principal, sent on the recipe routes |
| `FRAME_STORE_BUCKET`             | the materialisation bucket                       |

The identity it runs as needs to list and read objects in the bucket. `--write` also creates `mints/v5/` objects, so it needs object create.

From a workstation, set `DFO_APIS_URL=http://localhost:8083` and run `just apis-proxy` beside it, since the job sends a Cloud Run ID token only on Cloud Run, in a service (`K_SERVICE` set) or a job (`CLOUD_RUN_JOB` set).

```
uv run --package harness python -m harness.recipe_backfill               # dry run, the default
uv run --package harness python -m harness.recipe_backfill --write
uv run --package harness python -m harness.recipe_backfill --write --since 2026-09-14 --until 2026-09-21
uv run --package harness python -m harness.recipe_backfill --limit 50
```

A dry run builds every recipe and compares it with what apis holds. It sends no `PUT` and writes nothing to the bucket. `--write` puts each recipe to `PUT /api/datasets/frames/{spec_id}/{content_version}/recipe` and writes each recovered mint record to the bucket. `--since` and `--until` bound the frames by the time their object was written, and `--limit` takes the oldest N in range, so the job can run in parts. A run can be repeated: apis keeps a pair's first recipe, and a second `--write` run reports `already_stored` for each.

The job walks the frames oldest first and works on four at a time. For each derived frame it downloads the frame's object once (at most 64 MiB), checks its digest, and reads the trace from the Parquet footer. It decodes no rows, writes no frame object, and deletes nothing.

The report lists every derived frame with one reason:

| Reason               | Meaning                                                                             | Detail                                                     |
| -------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `stored`             | this run stored the recipe                                                          |                                                            |
| `already_stored`     | apis holds the recipe, with the same two hashes when it could be rebuilt            | `not rebuilt: <reason> ...` when it could not              |
| `would_store`        | dry run only: the recipe was built and apis holds none                              |                                                            |
| `not_published`      | apis has no version of the frame (404 `snapshot_not_found`), so no card draws on it |                                                            |
| `missing_ancestor`   | a frame it was computed from is gone from the bucket                                | the missing pair                                           |
| `root_unrecoverable` | a root dataset frame has no mint record and none could be rebuilt                   | the root pair and the reason, such as `no_version_matches` |
| `trace_unusable`     | a stored trace on the way yields no op-spec                                         | the pair and why                                           |
| `anchor_unresolved`  | a rolling frame's latest period cannot be stated from what the bucket holds         | the pair and the reason, such as `no_anchor_record`        |
| `conflict`           | apis refused the recipe (409 `publish_conflict`)                                    | the `constraint_id`                                        |
| `apis_failure`       | an apis service call failed after the client's retries                              | `status=` and `code=`                                      |
| `unexpected_error`   | reading the frame raised; the traceback is logged                                   | the exception type                                         |

Exit status is 0 when every frame ended on a settled reason, and 1 when any ended `apis_failure` or `unexpected_error`, which a rerun may clear.

The report closes with the numbers to read before the age rule returns:

```
Derived frames with a recipe stored in apis: 3 (stored 3, already stored 0)
Derived frames without a stored recipe: 3
  missing_ancestor   1
  root_unrecoverable 1
  conflict           1
  (2 of these had no recipe to send, so apis was not asked whether they are published)
Not published in apis, so no card draws on them: 1
Oldest frame object: 23.0 days (written 2026-09-14T00:01:00Z)
```

The first number is the frames whose cards can be refreshed after the bucket's objects are deleted. Every frame under the second is listed above with its reason, and a card drawn from one of them cannot be refreshed. A `conflict` whose detail is `recipe_requires_matching_normal_form_hash` or `recipe_requires_matching_replay_key` means apis holds another recipe for the pair. The last line is the age of the oldest object the walk saw: a restored 30-day rule deletes every object older than 30 days on its next pass.

## Conversation import

`python -m harness.adk_import` copies every saved Studio conversation from ADK's ledger sessions (app `dfo_analyst_graph.ledger` in the `sessions` table) into the harness's own tables (`harness/db/migrations`). The release that reads conversations from those tables copies a conversation itself when it reads one whose ADK ledger session moved since the job last copied it, so a turn the previous revision serves after the job ran is in the tables before the release's next request on that conversation reads them (`harness.adk_import.reimport_on_access`). That check is one lookup of ADK's `sessions` by its primary key per read, and the copy runs only when the session moved since it was last read. A copy that fails on a database fault fails the request before it writes anything, and the client's retry copies again. A conversation changed on both sides since the copy is diverged: the release answers every request on it with a 409 `conversation_diverged` refusal, which the client does not retry, and logs an error naming it and the `--prefer-adk` command that repairs it. A conversation whose ledger session the job would refuse is served as the tables hold it, and logged as an error once per ADK write on each instance; when the tables hold none of it, its next turn starts it afresh. The job copies every conversation in bulk before the release takes traffic, and its report and `--verify` are the evidence that every conversation opens as it did. The module docstring describes what is copied, what refuses a conversation and how a rerun decides.

The job reads the database from the settings the service uses: `DATABASE_URL`, or `SESSION_STORE_INSTANCE_CONNECTION_NAME`, `SESSION_STORE_DATABASE` and `SESSION_STORE_RUNTIME_DB_USER`. On dev and production the only database user is the `ai` service's runtime service account, connecting over IAM auth, and the instance is reachable only through the service's PSC network, so the job runs as a Cloud Run job execution under that service account, on that network, on the release's image. A workstation cannot reach it. The job reads ADK's tables by SQL and exits 2 when `adk_internal_metadata` names a schema version other than 1, when no database is configured, when a migration file is refused, or when `--verify` or `close-open-turns` finds the harness's tables missing. Only `import` applies the harness's migrations, so it can run before the release has started, and the other verbs run from an image older than the database's schema.

```
uv run --package harness python -m harness.adk_import import                  # dry run, the default
uv run --package harness python -m harness.adk_import import --apply
uv run --package harness python -m harness.adk_import import --verify
uv run --package harness python -m harness.adk_import import --apply --prefer-adk PRINCIPAL/CONVERSATION_ID ...
uv run --package harness python -m harness.adk_import import --apply --prefer-adk PRINCIPAL/CONVERSATION_ID --force
uv run --package harness python -m harness.adk_import close-open-turns         # dry run
uv run --package harness python -m harness.adk_import close-open-turns --apply
```

The import report lists each conversation as imported, re-imported, unchanged, in conflict or refused, by `principal/conversation_id`. A refusal names every structural fault of the conversation (a key the job does not know, a gap in the turn or approval positions, a run-ledger item naming a run the ledger lacks) or the error that stopped its write, and writes none of it. The job goes on with the rest and exits 1. The dry run builds every row the write would write, without locking or writing anything, so it reports the same refusals `--apply` would. Records the load from ADK skips or fails on, such as a plan record its model refuses or a spec block whose `_bound_refs` does not convert, are listed under "copied as stored" and copied verbatim. Under `--apply` a conflict marks the conversation diverged, and the release refuses it until `--prefer-adk` re-imports it. `--prefer-adk` refuses a conversation whose lease a stream holds, since taking ADK's copy ends that stream, unless `--force` is added; a name that matches no ledger session stops the job with exit status 2 before it writes anything.

Every spec block is copied with its `_bound_refs` as the list of `BoundEntity` rows `compiler.sources.source_from_spec_block` reads. A release before `BoundEntity` stored a mapping from entity text to refs, which that reader refuses, so a conversation's earlier sources would lose their entity refs on its next turn. The job converts each mapping as the module's `_convert_block` describes, and keeps a converted block only when the reader reads it. The report counts the blocks converted and those that do not convert in the ledger sessions, and on a line of its own those in the open turns' own sessions, which the release reads for a turn paused on ADK; an open turn's block that does not convert is listed under "open turn blocks the release reads as stored", since the job does not copy it and the release reads it from the turn session as it stands. The release reads every other open turn block through the same conversion (`harness.adk_import.open_turn_specs`), when it loads the turn and when a resumed stream carries the turn session's blocks, so a turn paused on ADK closes with blocks the reader reads. The job reads each open turn's session inside the conversation's own transaction, so a failure there refuses the conversation with nothing written. It also gives the p50, p95 and maximum size, in bytes of JSON text, of the ledger states and of the open turns' `orchestrator:specs`.

`--verify` compares, for every ledger session, the conversation as ADK's state loads, its blocks converted the same way, with `StudioStore.load` from the tables, canvas and run-ledger items included. Both sides read an open turn's blocks through the reader the release serves, and neither compares the lease's cancel flag, which a Stop sets on a copied lease. It lists each conversation that differs with the keys that differ. It lists apart, without comparing them, each conversation whose ADK state moved since the import while its rows did not ("moved on ADK since the import; re-imported on access": the previous revision, which serves until traffic moves, wrote it, and the release copies it on its next read), each conversation whose rows the serving release has written since the import, one written while verify compares it included, and each conversation in conflict. It ends with "N of M equal" over the rest, and exits 1 on a difference or a conflict.

### Cutover

`dfo-dev-ai` routes traffic to its latest revision, so the merge's deploy would put the release on traffic before the job has run. Traffic is pinned to the serving revision first, the release deploys beside it with no traffic, the job copies every conversation on the release's image, and traffic moves. The same steps run on dev when the release merges to `dev`, then on production when `dev` is released to `main` (`deploy-prod.yml` deploys every push to `main` the same way).

Hold any dfo-infra apply that touches the `ai` service from step 1 until step 5. Its Terraform traffic block (`latestRevision: true`) sends traffic to the latest revision, so an apply in that window moves traffic to the release early.

Set the environment's values. For dev:

```
REGION=$(gh variable get REGION --env dev)
PROJECT_ID=$(gh variable get PROJECT_ID --env dev)
SERVICE_AI=$(gh variable get SERVICE_AI --env dev)
```

For production, the same with `--env production`.

1. Before the merge, pin traffic to the revision serving now. The `jq` step reads the revision from the service's traffic and refuses unless one revision holds all of it. Settle a split or a rollback first:

   ```
   SERVING=$(gcloud run services describe "$SERVICE_AI" --region="$REGION" --project="$PROJECT_ID" --format=json \
     | jq -er '[.status.traffic[] | select((.percent // 0) > 0)]
       | if length == 1 and .[0].percent == 100 then .[0].revisionName
         else error("traffic is not on one revision: \(map({revisionName, percent}))") end') \
   && gcloud run services update-traffic "$SERVICE_AI" --region="$REGION" --project="$PROJECT_ID" \
     --to-revisions="$SERVING=100"
   ```

1. Merge, and wait for the deploy workflow to finish. The new revision starts with no traffic and applies the migrations. Confirm it took none: `gcloud run services describe "$SERVICE_AI" --region="$REGION" --project="$PROJECT_ID" --format='value(status.traffic)'` shows only `$SERVING`.

1. Create the job on the new revision's image, with the service's identity, network and database settings. The `jq` step refuses unless the service holds exactly the three `SESSION_STORE_*` variables, each as a literal value, and the job is created only when it succeeds:

   ```
   IMAGE=$(gcloud run services describe "$SERVICE_AI" --region="$REGION" --project="$PROJECT_ID" \
     --format='value(spec.template.spec.containers[0].image)')
   SA=$(gcloud run services describe "$SERVICE_AI" --region="$REGION" --project="$PROJECT_ID" \
     --format='value(spec.template.spec.serviceAccountName)')
   ENV=$(gcloud run services describe "$SERVICE_AI" --region="$REGION" --project="$PROJECT_ID" --format=json \
     | jq -er '[(.spec.template.spec.containers[0].env // [])[] | select(.name | startswith("SESSION_STORE_"))]
       | if length != 3 then error("the service holds \(length) SESSION_STORE_* variables, not 3")
         elif any(has("valueFrom") or (.value | type) != "string") then error("a SESSION_STORE_* variable is not a literal value")
         else map("\(.name)=\(.value)") | join(",") end') \
   && gcloud run jobs deploy "$SERVICE_AI-adk-import" --image="$IMAGE" --region="$REGION" --project="$PROJECT_ID" \
     --service-account="$SA" --network="$SERVICE_AI-psc" --subnet="$SERVICE_AI-psc-subnet" \
     --vpc-egress=private-ranges-only --set-env-vars="$ENV" --max-retries=0 --task-timeout=1800s \
     --command=python --args=-m,harness.adk_import,import
   ```

1. Execute the dry run and read its report in the execution's log. Cutover requires zero refusals. Fix each one before going on. A refusal left in place means that conversation starts afresh, since the release serves it as the tables hold it and they hold none of it; one that first appears after the dry run is repaired by fixing ADK's state and running `--prefer-adk` on it, which drops the turns the release wrote meanwhile. Then execute `--apply`, a bulk copy, so the release's first reads find every conversation in the tables and copy only what the previous revision wrote since, and then `--verify`. Its "N of N equal" is the acceptance evidence that every conversation opens as it did: record it. The previous revision keeps writing ADK until traffic moves, so verify lists the conversations it wrote since the copy as "moved on ADK since the import; re-imported on access" and leaves them out of the count. They are not differences.

   ```
   gcloud run jobs execute "$SERVICE_AI-adk-import" --region="$REGION" --project="$PROJECT_ID" --wait
   gcloud run jobs execute "$SERVICE_AI-adk-import" --region="$REGION" --project="$PROJECT_ID" --wait \
     --args=-m,harness.adk_import,import,--apply
   gcloud run jobs execute "$SERVICE_AI-adk-import" --region="$REGION" --project="$PROJECT_ID" --wait \
     --args=-m,harness.adk_import,import,--verify
   ```

1. Move traffic to the release, and let it follow the latest revision again:

   ```
   gcloud run services update-traffic "$SERVICE_AI" --region="$REGION" --project="$PROJECT_ID" --to-latest
   ```

1. Wait 15 minutes, Cloud Run's 900-second request timeout, so that no request of the previous revision is still writing ADK sessions. Then execute `--apply` and `--verify`. Their exit status 0 is the gate. The rerun copies each conversation the previous revision wrote that the release has not read since, and verify finds no difference and no conflict. Verify lists the conversations the release has written as "written by the tables' release since the import" and compares only the rest, so after a cutover with traffic its count is near "0 of 0 equal". The count recorded in the previous step is the acceptance evidence.

   ```
   gcloud run jobs execute "$SERVICE_AI-adk-import" --region="$REGION" --project="$PROJECT_ID" --wait \
     --args=-m,harness.adk_import,import,--apply
   gcloud run jobs execute "$SERVICE_AI-adk-import" --region="$REGION" --project="$PROJECT_ID" --wait \
     --args=-m,harness.adk_import,import,--verify
   ```

   Conflicts are expected to be few. The release copies a conversation from ADK before it serves any request on it, so every write it makes lands on a copy that holds every turn the previous revision had written by then. Two sources remain. One is a canvas edit served by the release while a stream on the previous revision runs: the release's read copied that stream's live lease, so no turn can start, but a canvas edit lands under the row lock, and the release's next copy after the stream writes again finds both sides changed. The other is a `/turn` that reaches both revisions at once and the release creates the turn session first: the previous revision answers a retryable 409 and leaves its claim on ADK beside the release's turn. When the previous revision creates it first, the release meets that session before it claims anything and answers a retryable 409; the retry re-imports the previous revision's turn and answers 409 `transient_unavailable` while that turn streams, with no conflict. A Stop causes no conflict. The previous revision's stream reads its cancel flag from ADK, so a Stop sent to the release does not reach it, and the flag it sets on the copied lease is left out of the comparison. The release logs each conflict it meets as an error naming the conversation and the command below, refuses every request on it with 409 `conversation_diverged`, and both `--apply` and `--verify` list it and exit 1. The conflict shows as `conversation_diverged` and is resolved with `--prefer-adk`. Given the name the report prints, it takes ADK's copy, which drops everything the release wrote for that conversation (its turns, canvas edits, approvals and runs) and serves the conversation again. It refuses while a stream holds the conversation's lease. Rerun it once the stream has ended.

   A rollback to the previous revision after this step loses what the release wrote, and rolling forward again turns every conversation both revisions served into a conflict: the post-drain `--apply` marks each one diverged and the release refuses it. Take the conflict list from that `--apply` report and pass every name to one `--prefer-adk` run (`--prefer-adk` takes a list), which takes ADK's copy of each.

1. Before the release that removes ADK from the turn path takes traffic, execute `--apply` and then `--verify` (the two commands of the previous step) and require exit status 0 from both, with no conflict. That release serves a diverged conversation from the tables and loses the previous revision's turns, and `--apply` copies any conversation ADK moved since the last copy. Then execute `close-open-turns` from this release's image, as a dry run and then with `--apply`, and once more after that release's own drain:

   ```
   gcloud run jobs execute "$SERVICE_AI-adk-import" --region="$REGION" --project="$PROJECT_ID" --wait \
     --args=-m,harness.adk_import,close-open-turns
   gcloud run jobs execute "$SERVICE_AI-adk-import" --region="$REGION" --project="$PROJECT_ID" --wait \
     --args=-m,harness.adk_import,close-open-turns,--apply
   ```

   It closes every turn left open on a pause, with no stream holding its lease, as `abandoned`, the way the next turn would have closed it, keeping the blocks of the sources its plans used from the turn's ADK session, converted as the import converts them. A turn a stream still holds and one with no ADK turn session (started by a release that runs turns without ADK) are listed and left open. A diverged conversation is listed and left alone, and the run exits 1. Run `--prefer-adk` on it first. A block that does not convert is kept as stored, as the import keeps one, and listed.

1. Delete the job: `gcloud run jobs delete "$SERVICE_AI-adk-import" --region="$REGION" --project="$PROJECT_ID" --quiet`.

The person running these needs to change the service's traffic, create and execute Cloud Run jobs in the project (`roles/run.developer`), and act as the runtime service account (`roles/iam.serviceAccountUser` on it). The job's own identity needs nothing the service does not already hold.

## Langfuse export

`python -m harness.langfuse_export` loads the Langfuse observations of one environment into the BigQuery table `observations`, day-partitioned on `start_time`. It runs as the Cloud Run job named by `JOB_LANGFUSE_EXPORT` on this image, and `build-deploy.yml` points that job at each new image in an Environment that sets the var. It reads these settings from the environment or `.env` and exits 2 naming any that is unset:

| Setting                       | Use                                               |
| ----------------------------- | ------------------------------------------------- |
| `LANGFUSE_PUBLIC_KEY`         | the Langfuse project's public key                 |
| `LANGFUSE_SECRET_KEY`         | its secret key                                    |
| `LANGFUSE_BASE_URL`           | the Langfuse origin, `https://cloud.langfuse.com` |
| `LANGFUSE_EXPORT_ENVIRONMENT` | the Langfuse environment to export, `dfo-prod-ai` |
| `EXPORT_BQ_PROJECT`           | the BigQuery project holding the dataset          |
| `EXPORT_BQ_DATASET`           | the dataset, `langfuse_export`                    |

Each run replaces every day from the latest loaded day minus one through today (UTC), and every day from 2026-10-05 on whose Langfuse observation count differs from its partition's row count, one load job per day writing that day's partition with `WRITE_TRUNCATE`. The counts come from `GET /api/public/v2/metrics` queries grouped by day, one for each 900 days since 2026-10-05, each asking for Langfuse's maximum of 1,000 rows; a query that answers 1,000 rows may be cut and fails the run. The day before the latest loaded day is read every run because Langfuse updates an observation's end time, usage and cost in place, which changes no count. A day's rows are keyed by observation id, so an observation two pages both hold loads once, as the copy with an end time, or the later-seen copy when neither or both have one. A replace never shrinks a partition: a day that reads fewer rows than its partition holds, none included, keeps the partition and prints both counts. A day that reads fewer rows than its partition holds or than Langfuse counts fails the run: the other days still load, then the run exits 1 naming every such day with its counts, and the day stays in later runs until the counts match. A day whose partition holds rows Langfuse no longer counts, such as an observation deleted in the Langfuse UI, fails every run until its rows are deleted from the table, after which the next run reloads it: `` bq query --use_legacy_sql=false 'DELETE FROM `PROJECT.DATASET.observations` WHERE DATE(start_time) = "YYYY-MM-DD"' ``. When the count query fails, the run still loads the days from the latest loaded day minus one through today under the never-shrink rule, then exits 1 with the query's error, which also names any day that read fewer rows than its partition holds. When every day of the run reads no rows, the run exits 1 naming the days and `LANGFUSE_EXPORT_ENVIRONMENT`, whether or not the count query failed, since a wrong environment or keys for another project read as empty and prod has had turns every day since 2026-10-05. Each day prints a line as it is replaced or kept, and a failure prints the day it stopped on. An empty table is backfilled from 2026-10-05. The job creates the table when it is absent, and a load adds any column the table lacks, which rows loaded earlier hold as NULL until their day is replaced. Its identity needs `roles/bigquery.jobUser` on the project and `roles/bigquery.dataEditor` on the dataset.

The rows hold ids, trace and session ids, names, type, level, environment, model, start and end time, latency, time to first token, tokens and cost. `input_tokens` is Langfuse's `inputUsage`, which includes cached input, and `cached_input_tokens` is `usageDetails.input_cached_tokens`, NULL on an observation without that key. The job never requests the `io` or `metadata` field groups and leaves out `statusMessage` and `userId`. The dfo organisation's Langfuse Cloud Pro plan shows 3 years of data, which reaches back past 2026-10-05, and sets the request limits the job keeps under. Langfuse requests go out at most one every 0.1 seconds, 600 a minute, and observation pages count against the General API bucket of 1,000 requests a minute. Metrics queries draw from the separate Metrics API bucket of 500 queries an hour and make at most two attempts: each hourly run spends one query and at most two through 2029-03-22, and two queries and at most four through 2031-09-08, and each `--check-day` spends two to four. A request error ends the run, with these rules: a 429 waits when its `Retry-After`, in seconds or as an HTTP date, is at most 60 seconds, or backs off for 15, 30, then 60 seconds, one step for each 429 in a row, when the header is missing or unusable, and ends the run when it asks for longer, as an exhausted Metrics bucket can, or when the attempts run out; a 5xx or a transport error is retried with backoff; any other 4xx ends the run. A refused run ends well inside the job's 1800-second timeout. A failed request's error carries the first 500 characters of Langfuse's answer.

`--check-day DAY` reads one day and compares its row count, summed cost and summed cached input tokens with `GET /api/public/v2/metrics` for the same day and environment. It needs only the Langfuse settings, writes nothing, and exits 1 on a difference. When Langfuse refuses the `usageByType` measure that sums cached input tokens, the check fails naming that measure.

```
uv run --package harness python -m harness.langfuse_export --check-day 2026-10-20
```

`--from DAY` adds every day from DAY through today to the run's days, under the rules above, so a column added to the table reaches history. Days run oldest first. Nothing before 2026-10-05, the first day exported, is read. Run it after a release adds a column. On the production job:

```
gcloud run jobs execute "$JOB_LANGFUSE_EXPORT" --region="$REGION" --project="$PROJECT_ID" \
  --args=-m,harness.langfuse_export,--from,YYYY-MM-DD --wait
```

A `--from` run that exits non-zero has replaced the days it reached and left the others as they were. The hourly runs do not repair the others, since their counts already match, so rerun it from the first day its execution's log reports neither `replaced` nor `partition kept` for. Each day prints its line as it finishes, so the log shows how far the run got. Rerunning from the original DAY would replace the same leading days again and, for a run that outlived the job's 1800-second task timeout, stop at about the same day. Every replace is whole-day and never shrinks, so a rerun is safe. For a long backfill, give that execution a longer deadline with `--task-timeout`:

```
gcloud run jobs execute "$JOB_LANGFUSE_EXPORT" --region="$REGION" --project="$PROJECT_ID" \
  --task-timeout=7200s --args=-m,harness.langfuse_export,--from,FIRST-DAY-NOT-REACHED --wait
```

Start it a few minutes after the hour, because the hourly trigger's run shares Langfuse's request budget.
