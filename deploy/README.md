# Deploy

GitHub Actions in this repo builds the `ai` image (the harness plus the
data-mcp, analysis-mcp, and presentation-mcp FastMCP apps it mounts
in-process) and swaps it
onto an existing Cloud Run service. It never creates or reconfigures a
service. Dev deploys automatically on every push to `dev` that touches the
image's inputs, in the ONEcampaign org. Production deploys on every push to
`main` of the DevelopmentFinanceObservatory copy of this repo, and on manual
dispatch there. Both run the same steps from `build-deploy.yml`. The sibling
`dfo-infra` repo owns the Cloud Run service definition (service account,
scaling, secrets, env vars) that these workflows target.

The workspace services (harness, data-mcp, analysis-mcp,
presentation-mcp) deploy as ONE Cloud Run service. See docs/architecture.md for the full rationale.

Until dfo-infra defines the Cloud Run service and someone sets the repo
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

| Name           | Kind   | Example value                                                                               | Who provides it                                        |
| -------------- | ------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `WIF_PROVIDER` | secret | `projects/123456789/locations/global/workloadIdentityPools/github/providers/github-actions` | dfo-infra's handoff (WIF pool/provider)                |
| `WIF_SA`       | secret | `dfo-ai-deployer@dev-fin-observatory.iam.gserviceaccount.com`                               | dfo-infra's handoff (deployer service account)         |
| `REGION`       | var    | `northamerica-northeast1` (dev), `europe-west1` (production)                                | dfo-infra's handoff                                    |
| `PROJECT_ID`   | var    | `dev-fin-observatory` (dev); the prod project ID in production                              | dfo-infra's handoff                                    |
| `AR_REPO`      | var    | `datacommons`                                                                               | dfo-infra's handoff (Artifact Registry repo)           |
| `SERVICE_AI`   | var    | `dfo-dev-ai`                                                                                | dfo-infra's handoff, once the Cloud Run service exists |

The Cloud Run service itself also needs `STUDIO_API_TOKEN` mounted as a
secret (dfo-infra, in the service definition next to `GOOGLE_API_KEY`). It
is the shared secret the frontend's proxy presents on studio requests. A
service without it accepts any non-empty bearer token and logs
`STUDIO AUTH IS OPEN` at startup.

`MCP_API_TOKEN` needs no secret on this one-Cloud-Run-service deployment.
Left unset, the harness mints a token at startup and hands it to its own
`data-mcp` / `analysis-mcp` / `presentation-mcp` MCP clients (`common.auth`),
so the three mounts require a credential nothing outside the process ever
holds. Set it as a secret only once one of those services is split out to
its own Cloud Run service (see "Splitting a service out later" below).
`DATA_MCP_URL` / `ANALYSIS_MCP_URL` / `PRESENTATION_MCP_URL` then point off
loopback, and startup refuses that combination without a shared
`MCP_API_TOKEN` the split-out service also has.

Every workflow that installs or builds the workspace (`ci.yml` and
`build-deploy.yml`) also reads the repo variable `READER_APP_CLIENT_ID` and
the repo secret `READER_APP_PRIVATE_KEY`, the credentials of the read-only
`dfo-reader` GitHub App. Each run mints a token with `contents: read` on
`ONEcampaign/dfo-spec-kit` from them to fetch the private dfo-spec-kit git
dependencies. They are repository-level, so they serve every environment.

The service name value above is a placeholder for illustration. dfo-infra
names the actual service, and whatever they pick goes in this var verbatim.

dfo-infra sets every variable and secret in the Environment: `SERVICE_AI`,
`WIF_PROVIDER`, `WIF_SA`, `REGION`, `PROJECT_ID`, and `AR_REPO`. Run
`just ENV=dfo-dev handoff ai` from the dfo-infra repo for the `dev`
Environment, and `just ENV=dfo-prod handoff ai` for `production`. The
script also sets the deployment branch policy. The values stay reproducible
and the Environment has one writer.

## GCP prerequisites (dfo-infra)

Before the first real deploy, dfo-infra needs to set up, per environment:

- A Workload Identity Federation pool and provider trusting this repo's
  GitHub Actions OIDC issuer (no JSON service account keys).
- A deployer service account, impersonable through that provider, with:
  - `roles/run.developer` (update Cloud Run services)
  - `roles/artifactregistry.writer` (push images)
  - `roles/iam.serviceAccountUser` (act as the Cloud Run runtime SA, which is
    a dedicated identity on the `ai` service)
- ONE Cloud Run service for the `ai` image (the harness process that
  mounts data-mcp, analysis-mcp, and presentation-mcp in-process, per
  docs/architecture.md),
  including its runtime service account, scaling config, env vars, and
  secrets. This repo's workflows only ever update the `--image` on the
  service that already exists.
- Model credentials on that service. google-adk reads Gemini/Vertex
  configuration straight from the process environment (`GOOGLE_API_KEY`, or
  `GOOGLE_GENAI_USE_VERTEXAI` + `GOOGLE_CLOUD_PROJECT` +
  `GOOGLE_CLOUD_LOCATION`), not from this repo's settings classes, so they
  must be wired into the service definition as env vars/secrets.
- `DFO_APIS_URL` on that service, pointing at dfo-apis, the FastAPI app that
  serves both the Search and Datasets surfaces. The data-mcp component
  mounted in the harness reads it to reach that API and falls back to a
  `http://localhost:8083` default without it, so the deployed service can't
  search until it's set. The value depends on the dfo-apis dev deployment
  existing. The Datasets surface additionally needs dfo-apis's
  `DFO_DATASETS_SURFACE` flag set, since that gates whether it mounts at all.
- `PRESENTATION_PUBLIC_URL` on that service, set to the service's own public
  `https` URL plus `/mcp/presentation`. Every presentation view's payload
  names its theme stylesheet under it, and the renderer's CSP admits its
  origin. It has no default. The harness fails at startup, naming the
  variable, when it is unset or not a valid absolute `http(s)` URL with no
  query, fragment, or credentials, so a revision without it never serves
  views a browser would render unstyled.
- The Artifact Registry repo named by `AR_REPO` in each environment's
  project (`datacommons` in dev).

## Repo-side prerequisites

- A `production` GitHub Environment on the DevelopmentFinanceObservatory
  repo with `main` as its only deployment branch. The release gate is the
  approval a sync pull request into `main` needs. Neither environment takes
  any other protection rule: dfo-infra's `scripts/handoff.sh` refuses to run
  against an environment with required reviewers or a wait timer.
- The vars/secrets table above, populated per environment.
- Staging is unbuilt. dfo-infra has no staging Terraform tree or GCP
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
   one). That env override is the seam `harness.settings.Settings` was
   built with, so no code change is needed on the harness side.
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
