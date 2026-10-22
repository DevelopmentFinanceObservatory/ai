# Contributing

Issues and pull requests are welcome. The ONE Data team maintains this repository. For a large change, open an issue first so we can agree on the approach before you write it.

Please follow the [Code of Conduct](https://github.com/DevelopmentFinanceObservatory/.github/blob/main/CODE_OF_CONDUCT.md). Report security problems through the repository's private vulnerability reporting (Security tab, "Report a vulnerability"), and see the [security policy](https://github.com/DevelopmentFinanceObservatory/.github/blob/main/SECURITY.md). For questions, see [SUPPORT.md](https://github.com/DevelopmentFinanceObservatory/.github/blob/main/SUPPORT.md).

## Setup

You need [`just`](https://github.com/casey/just) and [`uv`](https://docs.astral.sh/uv/). The checks below run without ONE's cloud access.

```bash
uv sync --all-packages        # or: just sync
uv run pre-commit install
```

Plain `uv sync` only syncs the virtual workspace root, so pass `--all-packages` to install every member into the shared `.venv`. Starting the harness locally with `just dev` needs API keys and a running `apis` service. The [README](README.md#getting-started) and [maintainer notes](docs/development.md) cover both.

## Before you open a pull request

```bash
just check          # lint, type check, test: CI's lint and test jobs
just fix            # apply ruff fixes and formatting
just test           # the whole suite from the workspace root
just docker-build   # the Cloud Run image, as CI's docker job builds it
```

CI runs three jobs on every ready pull request: `lint`, `test`, and `docker`. `just check` covers the first two. Run `just docker-build` when you change `deploy/Dockerfile.ai` or a dependency that lands in the image. Each workspace member has a shortcut, such as `just test-harness` and `just test-data-mcp`. CI skips draft pull requests.

Pre-commit runs ruff, `ty`, codespell, actionlint, and mdformat. Ruff and `ty` settings are in the root `pyproject.toml`.

## Branches and commits

`dev` is the trunk. Branch from it and open pull requests against it. Commit subjects follow Conventional Commits (`feat:`, `fix:`, `docs:`, `ci:`, `build:`).

## What a pull request needs

- `just check` passes, and CI is green.
- New behaviour has tests, and a bug fix has a test that fails without it.
- Docs change with the code, including the README of the component you touched and `.env.example` for a new setting.
- Settings are `pydantic-settings` classes that subclass `common.settings.BaseServiceSettings`. Read configuration through them, not `os.getenv`, except for variables a third-party library reads itself, such as `GOOGLE_API_KEY` for google-adk.
- A change to the digest or planner-context text can move a committed baseline. [docs/development.md](docs/development.md) says which command regenerates it.

Tests run from an empty working directory, so the suite reads no local `.env`. `test_scripts/` is ignored by git; use it for one-off local probes. Only `services/*/tests` and `packages/*/tests` run under `just check` and in CI.

[docs/development.md](docs/development.md) holds maintainer notes: secrets, adding a config key, baselines, the google-adk upgrade, and deploys.
