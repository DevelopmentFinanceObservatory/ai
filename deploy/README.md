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

`DFO_CALLER_PRINCIPAL` is the secret apis shares with the website, and must equal apis' and the website's value. The website names each `/mcp` client in `x-dfo-caller` as `ip:<digest>`. data-mcp and analysis-mcp send the secret and that caller with each dfo-apis call, beside `x-dfo-surface: mcp`, the client's `User-Agent` as `x-dfo-user-agent` and the name in its `clientInfo` as `x-dfo-mcp-client` (`invalid` for a name outside printable ASCII without quotes or backslashes, up to 128 characters; absent when the call carries no `clientInfo`, as a 2025-11-25 client's tool calls do, where analytics falls back to `x-dfo-user-agent`), and presentation-mcp sends them with each card an external client creates or revises, so apis counts the client's own allowance. For a Studio call, whose caller is `conversation:<digest>`, data-mcp and analysis-mcp send the secret, the caller, and the turn's id as `x-dfo-turn`, with no surface and no user agent, as does the harness for the dfo-apis calls its graph makes in process. Work several requests share, such as a card's replay or a series-statistics fetch, names no caller: it sends the secret and `x-dfo-surface` as the surface that started it (`studio`, `website` for a share the website refreshes, or `mcp`), which dfo-apis logs with no caller, and sends nothing to the pin, export and card doors, which refuse a vouched request without a caller. Unset, none sends a caller, so calls are not attributed to a client, and every external client's cards spend the one allowance of the address apis sees for this service. Deploy this harness only to an environment whose dfo-apis accepts a `conversation:` caller beside the principal and logs deliveries by caller and turn. An earlier dfo-apis answers 400 `request_invalid` to every Studio analysis, `export_data` and pin call, since infra mounts `DFO_CALLER_PRINCIPAL` on this service in dev and prod and the harness then names the conversation on each of them.

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

This image reads saved Studio conversations from the harness's own tables (`harness/db/migrations`) only. A deployment whose serving revision stores conversations in ADK's session tables (app `dfo_analyst_graph.ledger` in the `sessions` table) copies them across with the `python -m harness.adk_import` job on the image of the revision that added the tables, and that image's `deploy/README.md` gives the cutover steps. Its cutover's import is complete in an environment before this image merges there: `import --apply` and `import --verify` exit 0 with no conversation in conflict, `gate` exits 0, the previous revision is deleted, and `gate` exits 0 again. The start that applies `0003_turns.sql` runs the gate in two checks, each naming the count and the first ten. The first refuses with SQLSTATE `DFR02` while any ledger session holds a live `studio:segment` lease, which means a stream of the revision before the import still runs (`harness.db.cutover_gates.LIVE_LEDGER_LEASE_GATE`, run in 0003's transaction just before it). A lease lapses 45 s after its stream's last renewal, so startup retries this check within its 120 s budget. When it still fails, delete that revision, as the cutover step of the release that added the conversation tables does, run that job's `import --apply` and then `gate` until `gate` exits 0, and restart: the stream wrote its ledger session, so without the import the next start refuses with `DFA02`. A ledger session whose `studio:segment` lease has no numeric `expires_at` refuses with SQLSTATE `DFA06`, which startup does not retry, since no check can tell whether a stream holds it. No stream holds such a lease once every revision of an image before the release that added the conversation tables is deleted and that release's drain check has passed (its step 7). Then deploy a revision with `LAPSED_LEDGER_LEASES` set to the names, comma-separated, which the check then passes over: `gcloud run services update "$SERVICE_AI" --region="$REGION" --project="$PROJECT_ID" --update-env-vars="^@^LAPSED_LEDGER_LEASES=PRINCIPAL/CONVERSATION_ID,..."`. The variable has no effect once 0003 is recorded. The second, in 0003 itself, refuses with SQLSTATE `DFA02` while any ledger session is uncleared, and startup does not retry it. Once 0003 is recorded neither check runs again. That job clears the second on the previous image: `import --apply` copies a session ADK moved since its last copy, `import --apply --prefer-adk PRINCIPAL/CONVERSATION_ID` takes ADK's copy of a diverged one, `import --apply --keep-tables PRINCIPAL/CONVERSATION_ID` keeps the tables' copy of one they hold, `import --apply --discard PRINCIPAL/CONVERSATION_ID` gives up a refused one they do not hold, and `gate` then confirms.

That job's `close-open-turns --apply` must have exited 0 before this image merges, while the release before this one, which applied `0003_turns.sql` and `0004_card_lineage.sql` and runs turns without ADK, held all traffic; that release's `deploy/README.md` gives the step. It runs again after the revision deletes below, once their drain check passes. This image applies `0005_rename_adk_tables.sql`, which archives `adk_ledger_imports` and ADK's tables, and from its first start every verb of the job but `gate --adk-archived` exits 4 (`adk_archive_adk_ledger_imports` resolves), so `close-open-turns` cannot run once it has started. A turn the tables record as open with no turn behind it is a turn the release that ran turns on ADK left paused. This image closes such a turn as `abandoned` on the conversation's next turn, with what the conversation holds, which lacks the source blocks `close-open-turns` keeps from the turn's ADK session. An answer this image receives to such a pause finds no turn to resume.

## Renaming the ADK tables

Migration `0005_rename_adk_tables.sql` renames ADK's `sessions`, `events`, `app_states`, `user_states` and `adk_internal_metadata` tables and the import job's `adk_ledger_imports` to `adk_archive_<name>`, in the schema each is in, and renames each table's indexes and owned sequences the same way. It drops nothing and changes no row. A renamed table keeps its rows, keys and foreign keys, and `adk_archive_events` still references `adk_archive_sessions`. The runtime user needs ownership of the tables and `CREATE` on their schema. A table is ADK's when it has the columns of ADK's tables, and tables resolve through `search_path`. Startup applies the migration, so deploying the image that carries it runs it.

No revision that reads ADK's tables may exist once the migration commits. An instance of such an image that starts afterwards recreates empty ADK tables and serves an empty conversation list. Its new rows then sit in neither the harness tables nor the archive, and its fresh tables stop `restore` below. A revision with no share of traffic can still start instances, since Cloud Run's session-affinity cookie takes precedence over traffic splitting, so the cutover below deletes every such revision before this image merges.

Each check below stops startup with its own SQLSTATE, which startup does not retry, and renames nothing:

- `DFA01`: a table of an ADK name lacks ADK's columns while another carries them. Rename the foreign table by hand, or take a backup and discard ADK's tables with `drop-adk --apply --backup BACKUP_ID`, as the archive job below describes. `0005_rename_adk_tables.sql`'s own message names `drop-adk --apply` alone; the job needs `--backup BACKUP_ID` with it. With no ADK table at all, tables of those names are left alone without a message.
- `DFA02`: a ledger session, a row of ADK's `sessions` of app `dfo_analyst_graph.ledger`, is neither imported nor kept as of its last ADK write. A session is imported when `adk_ledger_imports` holds its row with `source_updated_at` equal to the session's `update_time`, not diverged, with `imported_digest` set, and `conversations` holds the conversation. It is kept when that current row has `kept_tables` set. With no `adk_ledger_imports` or no `conversations` table, no session is cleared. The message gives the count and the first ten keys. A database that applies `0003_turns.sql` and this migration in one start refuses at `0003`'s check of the same predicate first, so at startup this one fires when `0003` was applied earlier and an ADK-reading revision wrote sessions since. After a manual `restore --apply` the migration stays recorded, so startup refuses with `DFA05` instead, and `DFA02` comes from the `archive --apply` that renames the restored tables again. Run the import job of the release that added the conversation tables: `import --apply` copies them, `import --apply --prefer-adk PRINCIPAL/CONVERSATION_ID` takes ADK's copy of a conflict, and `import --apply --keep-tables PRINCIPAL/CONVERSATION_ID` keeps the tables' copy of one they hold, and `import --apply --discard PRINCIPAL/CONVERSATION_ID` gives up a refused one they do not hold. Then run its `gate`, and restart once it exits 0. To discard ADK's conversations instead, take a backup and run `drop-adk --apply`, as the archive job below describes.
- `DFA03`: a table to rename is not owned by the user the migration runs as, or that user lacks `CREATE` on its schema. Run it as the owner or grant `CREATE`.
- `DFA04`: for a table, index or sequence to rename, `adk_archive_<name>` would pass PostgreSQL's 63-byte identifier limit, which would truncate it, or a relation or type of that name exists in the schema; or, for a table to rename, a relation named `adk_archive_<table>` resolves through `search_path` in any schema. Rename the object to a shorter name, or rename or drop the one in the way. After the migration each `adk_archive_<table>` therefore resolves to the table it archived, and `restore` gives each object its own name back.
- `DFA07`: a conversation holds a turn left open on ADK that the import job's `close-open-turns` closes: an open turn whose ADK turn session exists and that `turn_cursors` holds no row for (`harness.db.cutover_gates.OPEN_TURNS_ON_ADK_GATE`, run just before this migration and by `archive`). This migration would archive the session that holds the turn's source blocks. It fires on a start that applies `0003_turns.sql` and this migration together, and on one after a cutover whose `close-open-turns` never exited 0. Deploy the release before this one if it is not serving, and run its cutover steps until the guarded `close-open-turns --apply` exits 0, then restart. A database with no ADK history applies every file in one start.
- `DFA05`: after the migrations, every start of this image checks whether any of ADK's five tables resolves through `search_path` under its own name with ADK's columns, and refuses while one does, since an image that reads them may be serving. The message names each such table and whether its own `adk_archive_<name>` resolves beside it, whether the archive resolves, each table of an archive name without the archive's columns, and whether `adk_ledger_imports` resolves under its own name. The check guards startup only and renames nothing. An instance of this image already running keeps serving whatever the tables do later, but no new instance starts, so the service cannot scale out or replace an instance until the step below runs. Tables of those names without ADK's columns are another service's and are left alone. The step depends on what the message names:
  - A live table beside its own `adk_archive_<name>`, or with no archive, no table of an archive name and no `adk_ledger_imports`, as `drop-archive` leaves the database: an image that reads ADK's tables started after the archive and created the table afresh, so it holds what that image wrote since. Find and delete that revision first. Only `drop-adk` or a move out of `search_path` clears these rows, since the import job cannot read them. Once no revision that reads ADK's tables is left, run `drop-adk` as a dry run and read the count of conversations, the ledger sessions of `sessions`, which tells a few stray conversations from the whole history, take a backup, then run `drop-adk --apply --backup BACKUP_ID`, which loses those writes, and restart. To keep the rows for a recovery by hand, move each table out of `search_path` instead with `ALTER TABLE ... SET SCHEMA`, which carries its indexes, owned sequences and the `events` foreign key.

  - Live tables without their `adk_archive_<name>` and with `adk_ledger_imports` under its own name, as after a manual `restore --apply`: the live tables are ADK's restored history. Once no ADK-reading revision is left, run the import job of the release that added the conversation tables to clear `DFA02` (`import --apply`, `import --verify`, `gate`), run `archive --apply`, which reruns the migration's four checks and renames the tables, then restart.

    If the release that added the conversation tables deleted its import job, recreate it on the image the job ran on, the image reference its README says to record before the deletion, as `IMAGE`, with the service's identity, network and the three `SESSION_STORE_*` variables, which it reads into `SA` and `ENV` as that README's job-creation step does:

    ```
    gcloud run jobs deploy "$SERVICE_AI-adk-import" --image="${IMAGE:?set it to the recorded image}" --region="$REGION" --project="$PROJECT_ID" \
      --service-account="$SA" --network="$SERVICE_AI-psc" --subnet="$SERVICE_AI-psc-subnet" \
      --vpc-egress=private-ranges-only --set-env-vars="$ENV" --max-retries=0 --task-timeout=1800s \
      --command=python --args=-m,harness.adk_import,import
    ```

  - Any other mix, as after an archive table was renamed or dropped by hand, or a restore forced with `--allow-missing`: put each table back under one name, all archived or all restored, and follow the matching case. A restore forced with `--allow-missing` from an archive without `adk_ledger_imports` leaves the first case's state, live tables with no archive and no `adk_ledger_imports`, but its rows are ADK's restored history, not a stray revision's writes. With no import records the import job cannot clear them, so either move the tables out of `search_path` to keep them, or take a backup and run `drop-adk --apply --backup BACKUP_ID`, which discards ADK's copy; the harness tables hold every conversation as of the cutover.

### The archive job

`python -m harness.db.adk_archive` acts on these tables from this image. Each verb runs in one transaction and prints the statements it ran. The transaction takes the migrations' advisory lock and statement and lock timeouts, so a verb never runs beside a starting instance's migrations. Without `--apply` it rolls the transaction back, so a dry run checks everything the verb would do and changes nothing. It connects with the settings the service uses, `DATABASE_URL` or the three `SESSION_STORE_*` variables over IAM auth, and exits 0 when it ran or found nothing to do, 1 when it or the database refused, and 2 on a usage error, a drop with `--apply` and no numeric `--backup`, or with no or an invalid database configuration.

`gcloud run jobs execute --wait` exits 1 for every non-zero status of a job. The job's own status is its task's `status.lastAttemptResult.exitCode`, which this prints for the newest execution of the job named `JOB`, and prints nothing for 0:

```
gcloud run jobs executions tasks list --region="$REGION" --project="$PROJECT_ID" \
  --execution="$(gcloud run jobs executions list --job="${JOB:?set it to the job name}" --region="$REGION" --project="$PROJECT_ID" --limit=1 --format='value(metadata.name)')" \
  --format='value(status.lastAttemptResult.exitCode)'
```

```
uv run --package harness python -m harness.db.adk_archive archive        # dry run, the default
uv run --package harness python -m harness.db.adk_archive archive --apply
uv run --package harness python -m harness.db.adk_archive restore [--allow-missing] [--apply]
uv run --package harness python -m harness.db.adk_archive drop-adk [--apply --backup BACKUP_ID]
uv run --package harness python -m harness.db.adk_archive drop-archive [--apply --backup BACKUP_ID]
```

- `archive` runs `0005_rename_adk_tables.sql` itself, after the `DFA07` check startup runs before it, so its checks hold, and records the migration in `harness_schema_migrations` as applied when it is not recorded yet. A second run finds nothing to rename.
- `restore` is a manual recovery tool outside the supported rollback. It renames every `adk_archive_<name>` table back, with its indexes and owned sequences, for an image that reads ADK's tables, and must run before such an image starts so that the image finds its tables and creates none. The restored tables lack every write to the harness tables since the cutover to the release that added the conversation tables, and anything a harness-reading revision writes while serving. It leaves the migration recorded in `harness_schema_migrations`, so a start of this image beside the restored tables refuses with `DFA05` (the restored-history case above) until `archive --apply` renames them again. It prints which of ADK's five tables the archive holds and refuses when it holds anything but not all five and `adk_archive_adk_ledger_imports`, the import ledger alone included: an image that reads ADK's tables would create a missing one empty, and restored history without the ledger has no import records for the import job to clear it with. `--allow-missing` restores the rest anyway. With nothing archived it finds nothing to do, so a rerun after a restore changes nothing. It refuses before renaming anything when an `adk_archive_<name>` table lacks the columns of the table it archived, as another service's table earlier on `search_path` would, or when a name it would restore exists in the object's schema, or, for a table, resolves through `search_path`, as fresh tables an ADK-reading image created do. After renaming it checks that each restored name resolves to the restored table and rolls back otherwise.
- `drop-adk` drops ADK's five tables, each that resolves through `search_path` and has ADK's columns, which discards ADK's copy of the conversations. It prints the count of conversations, the ledger sessions of `sessions` (app `dfo_analyst_graph.ledger`), before the `DROP`, in a dry run too. The count is a snapshot taken before the `DROP`.
- `drop-archive` drops every `adk_archive_<name>` table, once ADK's copy is no longer needed. It refuses, and drops nothing, when a table of such a name lacks the columns of the table it would archive, as another service's table earlier on `search_path` would.

Both drops name every table in one `DROP TABLE` statement without `CASCADE`, so the foreign key from `events` to `sessions` resolves within the statement. When anything else depends on one of the tables, such as a view, PostgreSQL refuses the statement, the job drops nothing and exits 1 with PostgreSQL's error, which names the dependent object. Drop or change that object by hand, then run the verb again.

The drops cannot be undone, and Cloud SQL's automated backups keep only their retention window, so take an on-demand backup before each one. With `--apply` each drop exits 2 before it connects unless `--backup BACKUP_ID` gives that backup's numeric id, and prints it with the statements it ran. The job cannot read the backup list, so the command below is the check: it reads the instance from the service's `SESSION_STORE_INSTANCE_CONNECTION_NAME`, points the job at the service's image, since the job keeps the image it was created on and an image older than `--backup` rejects it, creates the backup and waits for it, and prints its id only once Cloud SQL reports it `SUCCESSFUL`. It needs `cloudsql.backupRuns.create` on the instance (`roles/cloudsql.editor`). Set `VERB` to `drop-adk` or `drop-archive`:

```
CONNECTION=$(gcloud run services describe "$SERVICE_AI" --region="$REGION" --project="$PROJECT_ID" --format=json \
  | jq -er '[(.spec.template.spec.containers[0].env // [])[] | select(.name == "SESSION_STORE_INSTANCE_CONNECTION_NAME") | .value]
    | if length == 1 and (.[0] | split(":") | length) == 3 then .[0] else error("no single instance connection name") end') \
&& SQL_PROJECT=${CONNECTION%%:*} && INSTANCE=${CONNECTION##*:} \
&& IMAGE=$(gcloud run services describe "$SERVICE_AI" --region="$REGION" --project="$PROJECT_ID" \
  --format='value(spec.template.spec.containers[0].image)') \
&& gcloud run jobs update "$SERVICE_AI-adk-archive" --image="$IMAGE" --region="$REGION" --project="$PROJECT_ID" \
&& DESCRIPTION="before ${VERB:?set it to drop-adk or drop-archive} $(date -u +%FT%TZ)" \
&& gcloud sql backups create --instance="$INSTANCE" --project="$SQL_PROJECT" --description="$DESCRIPTION" \
&& BACKUP_ID=$(gcloud sql backups list --instance="$INSTANCE" --project="$SQL_PROJECT" \
  --filter="description=\"$DESCRIPTION\" AND status=SUCCESSFUL" --format='value(id)') \
&& [ -n "$BACKUP_ID" ] && [ "$(printf '%s\n' "$BACKUP_ID" | wc -l)" -eq 1 ] \
&& echo "backup $BACKUP_ID" \
&& gcloud run jobs execute "$SERVICE_AI-adk-archive" --region="$REGION" --project="$PROJECT_ID" --wait \
  --args=-m,harness.db.adk_archive,"$VERB",--apply,--backup,"$BACKUP_ID"
```

On dev and production the database is reachable only from the service's PSC network, as the runtime service account, so the job runs as a Cloud Run job on this image with the service's identity, network and database settings, as the conversation import job does. A workstation cannot reach it.

### Cutover

Set the environment's values. For dev:

```
REGION=$(gh variable get REGION --env dev)
PROJECT_ID=$(gh variable get PROJECT_ID --env dev)
SERVICE_AI=$(gh variable get SERVICE_AI --env dev)
```

For production, the same with `--env production`.

Before this image merges in an environment, delete every revision of the service older than the oldest revision of the release before this one: the revisions of the release that added the conversation tables, which read ADK's tables, and any revision a rollback started. One at 0% traffic still serves a client whose session-affinity cookie routes it there. The release before this one reads the harness tables only, so its revisions stay. List the revisions, newest first:

```
gcloud run revisions list --service="$SERVICE_AI" --region="$REGION" --project="$PROJECT_ID" \
  --sort-by=~metadata.creationTimestamp --format='table(metadata.name,metadata.creationTimestamp)'
```

Set `KEEP_FROM` to the oldest revision of the release before this one. The command below deletes every revision of the service created before it, oldest first, and refuses when none exists or one of them holds traffic. It confirms that none is left and prints three values to set in the shell for the check: `DELETED`, the names it deleted; `DELETED_AT`, the time before the first delete; and `DRAINED_AT`, the last delete's time plus the longest timeout (`spec.timeoutSeconds`) among the deleted revisions. A rollback to a deleted revision is a redeploy of its image by digest:

```
(
  set -euo pipefail
  : "${KEEP_FROM:?set it to the oldest revision of the release before this one}"
  CUT=$(gcloud run revisions describe "$KEEP_FROM" --region="$REGION" --project="$PROJECT_ID" \
    --format='value(metadata.creationTimestamp)')
  SERVED=$(gcloud run services describe "$SERVICE_AI" --region="$REGION" --project="$PROJECT_ID" --format=json \
    | jq -c '[.status.traffic[] | select((.percent // 0) > 0) | .revisionName]')
  OLD=$(gcloud run revisions list --service="$SERVICE_AI" --region="$REGION" --project="$PROJECT_ID" --format=json \
    | jq -er --arg cut "$CUT" --argjson served "$SERVED" '[.[] | select(.metadata.creationTimestamp < $cut)]
      | sort_by(.metadata.creationTimestamp)
      | if length == 0 then error("no revision is older than \($cut)")
        elif any(.[]; .metadata.name as $n | $served | index($n)) then error("a revision to delete holds traffic")
        else .[] | "\(.metadata.name) \(.spec.timeoutSeconds // 300)" end')
  DELETED_AT=$(date -u +%FT%TZ)
  NAMES="" WAIT=0
  while read -r REV LIMIT; do
    gcloud run revisions delete "$REV" --region="$REGION" --project="$PROJECT_ID" --quiet
    NAMES="$NAMES $REV"
    if [ "$LIMIT" -gt "$WAIT" ]; then WAIT=$LIMIT; fi
  done <<<"$OLD"
  LAST=$(date -u +%FT%TZ)
  LEFT=$(gcloud run revisions list --service="$SERVICE_AI" --region="$REGION" --project="$PROJECT_ID" --format=json \
    | jq --arg cut "$CUT" '[.[] | select(.metadata.creationTimestamp < $cut)] | length')
  [ "$LEFT" -eq 0 ] || { echo "$LEFT revisions older than $KEEP_FROM remain" >&2; exit 1; }
  NAMES=${NAMES# }
  DRAINED_AT=$(jq -rn --arg t "$LAST" --argjson s "$WAIT" '$t | fromdateiso8601 + $s | todateiso8601')
  echo "deleted $(jq -rn --arg n "$NAMES" '$n | split(" ") | length') revisions; set these for the check:"
  printf 'DELETED="%s" DELETED_AT=%s DRAINED_AT=%s\n' "$NAMES" "$DELETED_AT" "$DRAINED_AT"
)
```

Then check that the deleted revisions drained. A request a revision held at its delete ends within that revision's timeout, so the check refuses until `DRAINED_AT`. It then requires Cloud Run's audit log to record the delete of each name in `DELETED`, and prints, for reference, the requests the deleted revisions logged after `DELETED_AT`. It needs `roles/logging.viewer` and `roles/run.viewer`:

```
(
  set -euo pipefail
  : "${DELETED:?set it to the names the delete printed}" "${DELETED_AT:?set it to the time the delete printed}" \
    "${DRAINED_AT:?set it to the time the delete printed}"
  [ "$(date -u +%s)" -ge "$(jq -rn --arg t "$DRAINED_AT" '$t | fromdateiso8601')" ] \
  || { echo "a request of a deleted revision can run until $DRAINED_AT; rerun then" >&2; exit 1; }
  AUDIT=$(gcloud logging read "logName=\"projects/$PROJECT_ID/logs/cloudaudit.googleapis.com%2Factivity\" AND protoPayload.methodName:\"DeleteRevision\" AND timestamp>=\"$(jq -rn --arg t "$DELETED_AT" '$t | fromdateiso8601 - 3600 | todateiso8601')\"" \
    --project="$PROJECT_ID" --format=json) \
  || { echo "gcloud logging read failed: check roles/logging.viewer and PROJECT_ID" >&2; exit 1; }
  jq -er --arg d "$DELETED" '($d | split(" ")) - [.[] | (.protoPayload.resourceName // "") | split("/") | last]
    | if length == 0 then "the audit log records every delete" else error("no audit log entry deletes \(join(", "))") end' <<<"$AUDIT"
  echo "requests the deleted revisions logged after the delete, for reference:"
  gcloud logging read "logName=\"projects/$PROJECT_ID/logs/run.googleapis.com%2Frequests\" AND resource.labels.revision_name=($(jq -rn --arg d "$DELETED" '$d | split(" ") | map("\"\(.)\"") | join(" OR ")')) AND receiveTimestamp>=\"$DELETED_AT\"" \
    --project="$PROJECT_ID" --format='value(receiveTimestamp,resource.labels.revision_name,httpRequest.status,httpRequest.requestUrl)' \
  || echo "the request log could not be read" >&2
)
```

Then run the guarded `close-open-turns --apply` of the import job of the release that added the conversation tables until it exits 0, as that release's `deploy/README.md` gives it, with `NEXT` set to the serving revision of the release before this one. Its exit 0 is the evidence that the deleted revisions left nothing open. It fails while a turn is live on ADK, which only a stream of a deleted revision still running could hold, while a ledger session holds a live `studio:segment` lease, and while a ledger session moved since its import, which `import --apply` then copies before the next run. Each path of the release that added the conversation tables that opens, pauses or resumes a turn on ADK commits with the conversation's lease, and so does each such path of the release before it, whose canvas edits and Stops move the ledger session's `update_time`.

Then deploy this image as usual: its startup applies the migration. In each environment, as soon as the release has started and taken traffic, create the job and execute the `restore` dry run. Its exit status 0 and the ADK tables it lists, all five, are the evidence that the archive holds every table a manual restore would need: `restore` refuses when the archive lacks any of them. Keep the job: a manual recovery uses it.

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

Each later verb is an execution of the same job with its own arguments, for example `--args=-m,harness.db.adk_archive,restore,--apply`; a drop runs through the backup command above. Read the output in the execution's log. Once a later release no longer carries the job's module, delete the job: `gcloud run jobs delete "$SERVICE_AI-adk-archive" --region="$REGION" --project="$PROJECT_ID" --quiet`.

### Rollback

The only supported rollback target is the release before this one. It reads conversations from the harness tables only, and reads neither ADK's tables nor `adk_ledger_imports`, so it serves with the archive in place. Its startup logs `0005_rename_adk_tables.sql` as applied by a later release and carries on. Moving traffic back to it needs no step here.

An image refuses to start with `MigrationChecksumError` when an applied file's bytes changed, which no release does, or when the database recorded a migration file the image lacks whose name sorts at or below the image's own newest file. This image also refuses with `DFA05` beside ADK's live tables (above). The image that added the conversation tables holds 0001 and 0002, the release before this one 0001 to 0004, and this image 0001 to 0005, and the checks that run before `0003_turns.sql` and `0005_rename_adk_tables.sql` (`harness.db.cutover_gates`) add no file. So every rollback starts:

| Rollback to                                    | Database              | Starts                             | Step                                                                                                 |
| ---------------------------------------------- | --------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------- |
| The release before this one                    | 0001 to 0005 recorded | yes, logs 0005                     | none                                                                                                 |
| An earlier revision of this image              | 0001 to 0005 recorded | yes                                | none                                                                                                 |
| The release that added the conversation tables | 0001 to 0005 recorded | yes, logs 0003 to 0005             | unsupported: it reads ADK's tables, which 0005 archived, so `restore --apply` must run first (above) |
| The revision before the import                 | any                   | yes, it runs no harness migrations | unsupported, as above                                                                                |

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
