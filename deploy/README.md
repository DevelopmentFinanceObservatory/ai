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
  key, `GOOGLE_API_KEY`, which google-genai reads from the process environment, so
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

This image reads saved Studio conversations from the harness's own tables (`harness/db/migrations`) only. A deployment whose serving revision stores conversations in ADK's session tables (app `dfo_analyst_graph.ledger` in the `sessions` table) copies them across with the `python -m harness.adk_import` job on the image of the revision that added the tables, and that image's `deploy/README.md` gives the cutover steps. Its cutover's import is complete in an environment before this image merges there: `import --apply` and `import --verify` exit 0 with no conversation in conflict, `gate` exits 0, the previous revision is deleted, and `gate` exits 0 again. This image's first migration (`0003_turns.sql`) runs the gate's predicate and refuses startup with SQLSTATE `DFA02` while any ledger session is uncleared, naming the count and the first ten. That job clears them on the previous image: `import --apply` copies a session ADK moved since its last copy, `import --apply --prefer-adk PRINCIPAL/CONVERSATION_ID` takes ADK's copy of a diverged one, `import --apply --keep-tables PRINCIPAL/CONVERSATION_ID` keeps the tables' copy, and `gate` then confirms. Once this image takes traffic, that job's `close-open-turns --apply` is rerun until it exits 0. It leaves open every turn this image opened, which the `turn_cursors` table holds a row for by conversation and `turn_index`. A turn the tables record as open with no turn of this image behind it is closed as `abandoned` on the conversation's next turn, with what the conversation holds.

## Renaming the ADK tables

Migration `0005_rename_adk_tables.sql` renames ADK's `sessions`, `events`, `app_states`, `user_states` and `adk_internal_metadata` tables and the import job's `adk_ledger_imports` to `adk_archive_<name>`, in the schema each is in, and renames each table's indexes and owned sequences the same way. It drops nothing and changes no row. A renamed table keeps its rows, keys and foreign keys, and `adk_archive_events` still references `adk_archive_sessions`. The runtime user needs ownership of the tables and `CREATE` on their schema. A table is ADK's when it has the columns of ADK's tables, and tables resolve through `search_path`. Startup applies the migration, so deploying the image that carries it runs it.

No revision that reads ADK's tables may exist once the migration commits. An instance of such an image that starts afterwards recreates empty ADK tables and serves an empty conversation list. Its new rows then sit in neither the harness tables nor the archive, and its fresh tables stop `restore` below. A revision with no share of traffic can still start instances, since Cloud Run's session-affinity cookie takes precedence over traffic splitting, so the cutover below deletes every such revision before this image merges.

Each check below stops startup at once with its own SQLSTATE and renames nothing:

- `DFA01`: a table of an ADK name lacks ADK's columns while another carries them. Rename the foreign table by hand, or discard ADK's tables with `drop-adk --apply`. With no ADK table at all, tables of those names are left alone without a message.
- `DFA02`: a ledger session, a row of ADK's `sessions` of app `dfo_analyst_graph.ledger`, is neither imported nor kept as of its last ADK write. A session is imported when `adk_ledger_imports` holds its row with `source_updated_at` equal to the session's `update_time`, not diverged, with `imported_digest` set, and `conversations` holds the conversation. It is kept when that current row has `kept_tables` set. With no `adk_ledger_imports` or no `conversations` table, no session is cleared. The message gives the count and the first ten keys. A database that applies `0003_turns.sql` and this migration in one start refuses at `0003`'s check of the same predicate first, so this one fires when `0003` was applied earlier and an ADK-reading revision wrote sessions since, as after a rollback and `restore --apply`. Run the import job of the image that holds it: `import --apply` copies them, `import --apply --prefer-adk PRINCIPAL/CONVERSATION_ID` takes ADK's copy of a conflict, and `import --apply --keep-tables PRINCIPAL/CONVERSATION_ID` keeps the tables' copy. Then run its `gate`, and restart once it exits 0. To discard ADK's conversations instead, run `drop-adk --apply`.
- `DFA03`: a table to rename is not owned by the user the migration runs as, or that user lacks `CREATE` on its schema. Run it as the owner or grant `CREATE`.
- `DFA04`: for a table, index or sequence to rename, `adk_archive_<name>` would pass PostgreSQL's 63-byte identifier limit, which would truncate it, or a relation or type of that name exists in the schema; or, for a table to rename, a relation named `adk_archive_<table>` resolves through `search_path` in any schema. Rename the object to a shorter name, or rename or drop the one in the way. After the migration each `adk_archive_<table>` therefore resolves to the table it archived, and `restore` gives each object its own name back.
- `DFA05`: after the migrations, every start of this image checks that none of ADK's five tables resolves through `search_path` under its own name with ADK's columns, and refuses while one does, naming each. This image never serves beside them, since an image that reads them may be serving, as after a rollback and `restore --apply`, or may have started after the archive and created them afresh. The check renames nothing. Once no revision that reads ADK's tables is left, run `archive --apply`, which reruns the migration's four checks and renames the tables, then restart. Tables of those names without ADK's columns are another service's and do not refuse.

### The archive job

`python -m harness.db.adk_archive` acts on these tables from this image. Each verb runs in one transaction and prints the statements it ran. The transaction takes the migrations' advisory lock and statement and lock timeouts, so a verb never runs beside a starting instance's migrations. Without `--apply` it rolls the transaction back, so a dry run checks everything the verb would do and changes nothing. It connects with the settings the service uses, `DATABASE_URL` or the three `SESSION_STORE_*` variables over IAM auth, and exits 0 when it ran or found nothing to do, 1 when it or the database refused, and 2 on a usage error or with no or an invalid database configuration.

```
uv run --package harness python -m harness.db.adk_archive archive        # dry run, the default
uv run --package harness python -m harness.db.adk_archive archive --apply
uv run --package harness python -m harness.db.adk_archive restore [--allow-missing] [--apply]
uv run --package harness python -m harness.db.adk_archive drop-adk [--apply]
uv run --package harness python -m harness.db.adk_archive drop-archive [--apply]
```

- `archive` runs `0005_rename_adk_tables.sql` itself, so its four checks hold, and records the migration in `harness_schema_migrations` as applied when it is not recorded yet. A second run finds nothing to rename.
- `restore` renames every `adk_archive_<name>` table back, with its indexes and owned sequences. It leaves the migration recorded in `harness_schema_migrations`, so a start of this image beside the restored tables refuses with `DFA05` until `archive --apply` renames them again. It prints which of ADK's five tables the archive holds and refuses when it holds anything but not all five, the import ledger alone included, since an image that reads ADK's tables would create a missing one empty; `--allow-missing` restores the rest anyway. With nothing archived it finds nothing to do, so a rerun after a restore changes nothing. It refuses before renaming anything when an `adk_archive_<name>` table lacks the columns of the table it archived, as another service's table earlier on `search_path` would, or when a name it would restore exists in the object's schema, or, for a table, resolves through `search_path`, as fresh tables an ADK-reading image created do. After renaming it checks that each restored name resolves to the restored table and rolls back otherwise.
- `drop-adk` drops ADK's five tables, each that resolves through `search_path` and has ADK's columns, which discards ADK's copy of the conversations.
- `drop-archive` drops every `adk_archive_<name>` table, once ADK's copy is no longer needed. It refuses, and drops nothing, when a table of such a name lacks the columns of the table it would archive, as another service's table earlier on `search_path` would.

Both drops name every table in one `DROP TABLE` statement without `CASCADE`, so the foreign key from `events` to `sessions` resolves within the statement. When anything else depends on one of the tables, such as a view, PostgreSQL refuses the statement, the job drops nothing and exits 1 with PostgreSQL's error, which names the dependent object. Drop or change that object by hand, then run the verb again.

On dev and production the database is reachable only from the service's PSC network, as the runtime service account, so the job runs as a Cloud Run job on this image with the service's identity, network and database settings, as the conversation import job does. A workstation cannot reach it.

### Cutover

Set the environment's values. For dev:

```
REGION=$(gh variable get REGION --env dev)
PROJECT_ID=$(gh variable get PROJECT_ID --env dev)
SERVICE_AI=$(gh variable get SERVICE_AI --env dev)
```

For production, the same with `--env production`.

Before this image merges in an environment, delete every revision of the service older than the oldest revision of the release before this one: the revision of the release that added the conversation tables, which reads ADK's tables, and any revision a rollback started. The release before this one reads the harness tables only, so its revisions stay. List the revisions, newest first, and delete each older one by name:

```
gcloud run revisions list --service="$SERVICE_AI" --region="$REGION" --project="$PROJECT_ID" \
  --sort-by=~metadata.creationTimestamp --format='table(metadata.name,metadata.creationTimestamp)'
gcloud run revisions delete "${REVISION:?set it to a revision to delete}" \
  --region="$REGION" --project="$PROJECT_ID" --quiet
```

Then list them again: the list holds only revisions of the release before this one. Confirm that each deleted revision's instance count has reached zero, from Cloud Monitoring's `run.googleapis.com/container/instance_count`, since a deleted revision is gone from `gcloud run revisions describe`. Cloud Monitoring samples the metric every 60 s and can show a point up to about 180 s after its time, so wait at least 5 minutes after the delete before reading it. The command prints each series' newest points, newest first, by instance state. The revision has drained when every state's series has a point dated after the delete that reads 0, or when the command prints no series at all and its window covers at least 5 minutes after the delete:

```
START=$(date -u -d '-30 min' +%FT%TZ 2>/dev/null || date -u -v-30M +%FT%TZ)
curl -sG -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://monitoring.googleapis.com/v3/projects/$PROJECT_ID/timeSeries" \
  --data-urlencode "filter=metric.type=\"run.googleapis.com/container/instance_count\" AND resource.labels.revision_name=\"$REVISION\"" \
  --data-urlencode "interval.startTime=$START" \
  --data-urlencode "interval.endTime=$(date -u +%FT%TZ)" \
  | jq -r '.timeSeries[]? | .metric.labels.state as $state
    | .points[:5][] | "\(.interval.endTime) \($state) \(.value.int64Value)"'
```

Then deploy this image as usual: its startup applies the migration. In each environment, as soon as the release has started and taken traffic, create the job and execute the `restore` dry run. Its exit status 0 and the ADK tables it lists, all five, are the evidence that a rollback to an image that reads ADK's tables would find every table it needs: `restore` refuses when the archive lacks any of them. Keep the job: a rollback uses it.

Create the job on the serving image. The `jq` step refuses unless the service holds exactly the three `SESSION_STORE_*` variables, each as a literal value, and the job is created only when it succeeds:

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
&& gcloud run jobs deploy "$SERVICE_AI-adk-archive" --image="$IMAGE" --region="$REGION" --project="$PROJECT_ID" \
  --service-account="$SA" --network="$SERVICE_AI-psc" --subnet="$SERVICE_AI-psc-subnet" \
  --vpc-egress=private-ranges-only --set-env-vars="$ENV" --max-retries=0 --task-timeout=600s \
  --command=python --args=-m,harness.db.adk_archive,restore
gcloud run jobs execute "$SERVICE_AI-adk-archive" --region="$REGION" --project="$PROJECT_ID" --wait
```

Each later verb is an execution of the same job with its own arguments, for example `--args=-m,harness.db.adk_archive,drop-archive,--apply`. Read the output in the execution's log. Once a later release no longer carries the job's module, delete the job: `gcloud run jobs delete "$SERVICE_AI-adk-archive" --region="$REGION" --project="$PROJECT_ID" --quiet`.

### Rollback

The ordinary rollback target is the release before this one. It reads conversations from the harness tables only, and reads neither ADK's tables nor `adk_ledger_imports`, so it serves with the archive in place. Its startup logs `0005_rename_adk_tables.sql` as applied by a later release and carries on. Moving traffic back to it needs no step here.

To roll back further, to an image that reads ADK's tables, run `restore` with `--apply` before that image starts, so it finds its tables and creates none. When an image that reads ADK's tables has already started and created fresh tables, `restore` refuses and names them. The fresh tables then hold the only copy of what that image wrote since. Run `drop-adk --apply` to discard them, then `restore --apply`:

```
gcloud run jobs execute "$SERVICE_AI-adk-archive" --region="$REGION" --project="$PROJECT_ID" --wait \
  --args=-m,harness.db.adk_archive,restore,--apply
```

`restore --apply` leaves the migration recorded, so an instance of this image that starts beside the restored tables, during the rollback or after it, refuses with `DFA05`, fails its startup probe and renames nothing. A roll forward from an image that reads ADK's tables passes through the release before this one, and the `archive` verb renames the tables, with the migration's checks, `DFA02` included:

1. Run the import job of the release that added the conversation tables: its `import --apply` and `import --verify` until each exits 0 with no conversation in conflict, then its `gate` until it exits 0.
1. Move traffic to the release before this one. Delete every revision that reads ADK's tables, the one the rollback started included, and confirm each is gone and drained, with the commands of the cutover above. Run the `gate` again, and rerun the import and the `gate` until the `gate` exits 0.
1. Execute this image's job with `--args=-m,harness.db.adk_archive,archive,--apply`. It refuses with `DFA02` while a conversation the ADK-reading image wrote has not reached the harness tables; clear it with the import job, then run `archive --apply` again.
1. Deploy this image. Its startup finds the tables archived and proceeds.

To discard ADK's conversations instead of importing them, run `drop-adk --apply` after the deletes of the second step, in place of the import and the `gate`.

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
