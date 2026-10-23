# DFO AI services

The AI layer of the [Development Finance Observatory](https://github.com/DevelopmentFinanceObservatory) (DFO). It holds an agent harness that answers questions about development finance, and the MCP servers the harness calls to find data, run analysis, and draw charts and tables. The DFO is an initiative of [ONE Data](https://data.one.org), built on [Google Data Commons](https://datacommons.org). The organisation profile has the full map of repositories.

This repository is under active development. Interfaces, schemas, and data may change without notice.

## What's inside

A [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) monorepo. Each member has its own dependencies and tests, and all of them deploy together as one Cloud Run service.

```text
packages/common/            Shared logging, telemetry, settings base, and materialisation store
services/harness/           Agent loop on Google ADK and FastAPI, streamed over SSE
services/data-mcp/          Data discovery, minting, materialisation, and graph entity description
services/analysis-mcp/      Deterministic computation across seven analytical verbs
services/presentation-mcp/  Canvas component authoring and frame view tools
evals/                      ADK eval sets, run by hand against a live model
docs/                       Architecture notes, the studio interface, and the MCP user guide
deploy/                     Dockerfile and deploy notes
```

The harness also holds the `orchestrator`, `planner`, `compiler`, `translator`, and `executor` modules under `services/harness/src/`. Each service uses a src layout, `services/<name>/src/<package>/`, with tests in `services/<name>/tests/`.

The servers query the knowledge graph through the Datasets and Search APIs in [`apis`](https://github.com/DevelopmentFinanceObservatory/apis), and the public site in [`website`](https://github.com/DevelopmentFinanceObservatory/website) talks to the harness.

## Getting started

Running the whole stack needs access to ONE's Google Cloud project (`dev-fin-observatory`) for the Gemini and Langfuse keys, and a reachable `apis` deployment for data. Without that access you can read the code, run the lint, type checks, and tests, and read the docs.

Without cloud access you need [`just`](https://github.com/casey/just) and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --all-packages
just check        # lint, type check, and the full test suite
```

With access, authenticate and start the harness on `localhost:8080`, with the three MCP servers mounted in-process:

```bash
gcloud auth login
gcloud auth application-default login
just setup        # generates .env, syncs dependencies, installs pre-commit hooks
just dev
```

`just` lists every recipe, and `just doctor` checks the local environment.

## Documentation

- [Architecture](docs/architecture.md) explains how the services fit together.
- [Studio interface](docs/frontend-interface.md) is the integration guide for frontend developers, and [the wire contract](docs/studio-contract-v1.md) records the protocol decisions behind it.
- [MCP user guide](docs/guide/connect.md) shows how to connect an AI assistant to the DFO server.
- [Maintainer notes](docs/development.md) cover configuration, secrets, and upgrade hazards, and [deploy/README.md](deploy/README.md) covers deploys.
- Each component has its own README: [common](packages/common/README.md), [harness](services/harness/README.md), [data-mcp](services/data-mcp/README.md), [analysis-mcp](services/analysis-mcp/README.md), and [presentation-mcp](services/presentation-mcp/README.md).

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

MIT. See [LICENSE](LICENSE).
