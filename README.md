# dfo-ai

The AI layer of the Development Finance Observatory (DFO). This repository holds the agent
harness, the MCP servers it calls, and their shared modules. It
ships independently of the knowledge graph, data pipelines, and
the public site.

This repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
monorepo. Each package (`harness`, `data-mcp`, `analysis-mcp`, `presentation-mcp`, and `packages/common`)
is the unit of code ownership, with its own dependencies and tests.
They deploy together as a single Cloud Run service.

## Repo map

| Repo       | What it is                                                                                                                     |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `dcp`      | The knowledge graph engine (ONE's fork of Data Commons) that serves the graph over DCP mixer and SDMX APIs.                    |
| `apis`     | The Datasets API and Search API that `data-mcp` wraps to query the `dcp` graph.                                                |
| `ai`       | This repository, which houses the agent harness and MCP servers.                                                               |
| `spec-kit` | Shared spec models for spec-id hashing, materialisation, manifest schemas, and component specifications.                       |
| `website`  | The public Next.js site (chat and canvas workspace, reference pages, and the proxy for harness traffic).                       |
| `skills`   | The open repository of domain-knowledge packages (definitions, methodology, analytical recipes) loaded dynamically at runtime. |
| `infra`    | Infrastructure as code (Cloud Run service definitions, service accounts, scaling, and secrets).                                |

## Layout

```
ai/
  packages/
    common/                 # Shared logging, telemetry, settings base, and materialisation store
  services/
    harness/
      src/harness/            # Agentic loop on Google ADK and FastAPI with SSE event streaming
      src/orchestrator/       # The agent that answers a studio turn by calling its tools
      src/planner/            # Methodological planning engine gated behind user approval
      src/compiler/           # Builds and checks the workflow an approved plan runs
      src/translator/         # Finds each requirement's data and proposes the operations the compiler builds
      src/executor/           # Runs an approved workflow and checks each result against the plan
    data-mcp/                # Data discovery, minting, materialisation, and graph entity description
    analysis-mcp/            # Deterministic computation across seven op-spec analytical verbs
    presentation-mcp/        # Canvas component authoring and frame view tools for the web workspace
  evals/                    # ADK eval sets run manually against a live model
  docs/
    architecture.md
```

Each service follows a src layout, `services/<name>/src/<pkg>/`, with tests
under `services/<name>/tests/`. Package names are `harness`, `data_mcp`,
`analysis_mcp`, `presentation_mcp`, and `common`. `orchestrator`, `planner`,
`compiler`, `translator` and `executor` live inside `services/harness/src/` as
further top-level modules so they share harness dependencies, tests, and CI
pipelines during development. Their tests live under
`services/harness/tests/orchestrator/`, `services/harness/tests/planner/`,
`services/harness/tests/compiler/`, `services/harness/tests/translator/` and
`services/harness/tests/executor/`.

Detailed package documentation lives in each component directory:

- [`packages/common`](packages/common/README.md) covers shared configuration, telemetry, and materialisation storage.
- [`services/harness`](services/harness/README.md) covers the studio surface, the agent's workflow, sub-app mounting, and SSE transport. Frontend developers start at [`docs/frontend-interface.md`](docs/frontend-interface.md); the contract's design history is [`docs/studio-contract-v1.md`](docs/studio-contract-v1.md).
- [`services/data-mcp`](services/data-mcp/README.md) covers search discovery, dataset retrieval, and parquet minting.
- [`services/analysis-mcp`](services/analysis-mcp/README.md) covers analytical operations, op-spec hashing, and lineage.
- [`services/presentation-mcp`](services/presentation-mcp/README.md) covers canvas component generation and UI contracts.

## Quickstart

You need [`just`](https://github.com/casey/just), [`uv`](https://docs.astral.sh/uv/),
and access to the `dev-fin-observatory` GCP project.

```
gcloud auth login
gcloud auth application-default login
just setup
just dev
```

`just setup` checks prerequisites, pulls API keys from Secret Manager to generate `.env`,
syncs workspace members into the shared `.venv`, and installs pre-commit hooks.
`just dev` starts the harness on `localhost:8080` with `data-mcp`, `analysis-mcp`,
and `presentation-mcp` mounted in-process.

`just` lists all recipes, and `just doctor` checks the local environment.

`dev` is the trunk branch. Create feature branches from `dev` and open pull requests against it.
[CONTRIBUTING.md](./CONTRIBUTING.md) covers configuration, secrets management,
and developer commands.

## Deploys

GitHub Actions builds one container image (`ai`) containing the harness and
all three mounted MCP servers. Merges to `dev` deploy automatically to the Cloud Run
development environment. Production deploys on every push to `main`.
`infra` defines the Cloud Run service configuration (service accounts, scaling limits,
secrets). Repository CI only updates the container image deployed to that service.
