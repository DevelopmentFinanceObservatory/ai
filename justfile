# dfo-ai — Developer Commands
# Run 'just' or 'just --list' to see all available targets.
#
# Override defaults from the command line:
#   just PORT=9000 dev
#   just GCP_PROJECT=other-project env

# Passes recipe parameters to a shebang recipe's shell as "$1", "$2", ...
# instead of only via {{...}} text substitution. `ask` relies on this to
# forward its variadic ARGS through "$@" without ever re-interpreting the
# question text as shell.
set positional-arguments := true

# Configuration (override any of these from the command line)
ENV_FILE    := ".env"
ENV_SAMPLE  := ".env.example"
GCP_PROJECT := "dev-fin-observatory"

# Host port for the harness dev server. Also written into .env by `just env`.
PORT := "8080"

# The platform presentation-mcp talks to. `.env` carries no value and the
# service default is localhost:8083, where nothing listens, so every search
# fails silently without this.
DFO_APIS_URL := "https://dfo-dev-apis-zms5lj4zhq-nn.a.run.app"

# The deployed dev harness `ask-dev` targets, so asking it needs no --url
# typed by hand. Pinned as a literal rather than resolved with `gcloud run
# services describe` on every call: `ask-dev` is interactive and a Cloud Run
# URL is stable across ordinary deploys, only changing if the service is
# torn down and recreated, so a slow gcloud round trip on every question
# isn't worth paying -- update this line if that happens.
STUDIO_DEV_URL := "https://dfo-dev-ai-zms5lj4zhq-nn.a.run.app"

# Show available commands (default)
_default:
    @just --list --unsorted

# ── Setup & Configuration ────────────────────────────

# First-time setup: check prerequisites, create the env file, sync deps, install hooks
setup: _check-prerequisites env
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Syncing dependencies..."
    uv sync --all-packages
    echo ""
    # The hooks run `just typecheck`, which resolves imports against .venv -- so
    # the workspace has to be synced before the hooks are installed, not after.
    echo "Installing pre-commit hooks..."
    uv run pre-commit install
    echo ""
    echo "Setup complete. Run 'just dev' to start the harness, or 'just doctor' to check the environment."

# Create the local env file, filling secret-backed keys from Secret Manager
env:
    #!/usr/bin/env bash
    set -euo pipefail
    TARGET="{{ENV_FILE}}"
    SAMPLE="{{ENV_SAMPLE}}"
    if [ ! -f "$SAMPLE" ]; then
        echo "Error: $SAMPLE not found. Cannot create env file."
        exit 1
    fi
    if [ ! -f "$TARGET" ]; then
        echo "Copying $SAMPLE -> $TARGET..."
        cp "$SAMPLE" "$TARGET"
        chmod 600 "$TARGET"
        echo ""
    fi
    current() {
        grep -s "^${1}=" "$TARGET" | head -1 | cut -d= -f2-
    }
    write() {
        # Fill the empty KEY= line the sample ships, or append if there is none.
        #
        # Done in bash rather than with `sed -i`, for two reasons: a secret
        # passed as a sed argument is visible to `ps` for the life of the process,
        # and `sed -i.bak` leaves a plaintext copy of the file behind if the run is
        # interrupted between the edit and the cleanup. Here the value only ever
        # lives in a shell variable, and the rewrite lands atomically via mv.
        local key="$1" value="$2" tmp found=0
        tmp=$(mktemp "${TARGET}.XXXXXX")
        chmod 600 "$tmp"
        # `|| [ -n "$line" ]` so a final line with no trailing newline survives.
        while IFS= read -r line || [ -n "$line" ]; do
            if [ "$line" = "${key}=" ]; then
                printf '%s=%s\n' "$key" "$value" >> "$tmp"
                found=1
            else
                printf '%s\n' "$line" >> "$tmp"
            fi
        done < "$TARGET"
        [ "$found" -eq 1 ] || printf '%s=%s\n' "$key" "$value" >> "$tmp"
        mv "$tmp" "$TARGET"
    }
    # key:secret-name:requirement[:blank-message]. Anything not listed here
    # is a non-secret default that ships filled in .env.example and needs no
    # resolution. blank-message explains what staying unset costs for that
    # key specifically -- printing the Langfuse consequence for a key that
    # isn't Langfuse's would be wrong.
    resolve() {
        local key="$1" secret="$2" required="$3"
        local blank_message="${4:-tracing stays off until it is set}"
        local value
        if [ -n "$(current "$key")" ]; then
            echo "  $key already set."
            return
        fi
        # An exported shell variable wins over Secret Manager, so a one-off
        # override works without editing anything: GOOGLE_API_KEY=... just env
        # Prefix that with a space to keep the key out of your shell history.
        value="${!key:-}"
        if [ -n "$value" ]; then
            write "$key" "$value"
            echo "  $key taken from the environment."
            return
        fi
        value=$(just _secret "$secret")
        if [ -n "$value" ]; then
            write "$key" "$value"
            echo "  $key from Secret Manager ($secret)."
        elif [ "$required" = required ]; then
            read -rp "  $key (enter to skip): " value || true
            write "$key" "$value"
            if [ -n "$value" ]; then
                echo "  Set $key."
            else
                echo "  $key left blank -- the harness will not answer until it is set."
            fi
        else
            echo "  $key left blank ($blank_message)."
        fi
    }
    # Carry over keys added to the sample since this .env was written, with the
    # sample's default. This is what keeps a committed non-secret default --
    # like a new MODEL_NAME -- flowing to everyone on a git pull instead of
    # being announced by hand. Commented-out lines are excluded, so optional
    # keys stay opt-in.
    added=0
    while IFS= read -r line; do
        key="${line%%=*}"
        grep -q "^${key}=" "$TARGET" 2>/dev/null && continue
        printf '%s\n' "$line" >> "$TARGET"
        echo "  Added $key from $SAMPLE."
        added=$((added + 1))
    done < <(grep -E '^[A-Z_0-9]+=' "$SAMPLE")
    [ "$added" -gt 0 ] && echo ""

    echo "Checking environment variables..."
    resolve GOOGLE_API_KEY      dfo-dev-gemini-api-key       required
    resolve LANGFUSE_PUBLIC_KEY dfo-dev-langfuse-public-key  optional
    resolve LANGFUSE_SECRET_KEY dfo-dev-langfuse-secret-key  optional
    resolve STUDIO_API_TOKEN    dfo-dev-studio-api-token     optional "studio auth stays open to any non-empty token until it is set"
    resolve MCP_API_TOKEN       dfo-dev-mcp-api-token        optional "the harness mints a per-process token and every external MCP client gets a 401"
    resolve DATASETS_FIRST_PARTY_PRINCIPAL dfo-dev-datasets-first-party-principal optional "presentation-mcp cannot publish an analysis-minted frame, so an analysis verb answers in prose and renders no card"
    echo ""
    echo "$TARGET is ready. Run 'just dev' to start the harness."

# Prints the value to stdout (empty if it cannot be resolved); progress and hints
# go to stderr, so this composes as `value=$(just _secret <name>)`. When the exact
# name misses, it retries against the generated-suffix shape some secrets carry
# (dfo-dev-dc-google-maps-api-key-f1c75504), and refuses to guess between two.
#
# Read one secret from Secret Manager
[private]
_secret NAME:
    #!/usr/bin/env bash
    set -euo pipefail
    # stderr goes to a file, not /dev/null, so PERMISSION_DENIED can be told
    # apart from NOT_FOUND.
    err=$(mktemp)
    trap 'rm -f "$err"' EXIT
    if value=$(gcloud secrets versions access latest --secret="{{NAME}}" --project={{GCP_PROJECT}} 2>"$err"); then
        printf '%s' "$value"
        exit 0
    fi
    if grep -q PERMISSION_DENIED "$err"; then
        account=$(gcloud config get-value account 2>/dev/null || true)
        # Secret Manager returns PERMISSION_DENIED instead of NOT_FOUND on
        # purpose, so a caller without access can't probe which secrets exist.
        echo "  Access denied reading {{NAME}} in {{GCP_PROJECT}} as ${account:-the active account} -- a permissions problem, not a naming one." >&2
        echo "  -> ask an owner to run:" >&2
        echo "       gcloud secrets add-iam-policy-binding {{NAME}} --member='user:${account:-YOU@one.org}' --role='roles/secretmanager.secretAccessor' --project={{GCP_PROJECT}}" >&2
        exit 0
    fi
    echo "  {{NAME}} not found by exact name in {{GCP_PROJECT}}. Trying a suffixed match..." >&2
    if ! all=$(gcloud secrets list --project={{GCP_PROJECT}} --format="value(name)" 2>"$err"); then
        if grep -q PERMISSION_DENIED "$err"; then
            # secretAccessor grants versions.access, not secrets.list -- a
            # correctly-provisioned developer still can't list.
            echo "  Can't list secrets in {{GCP_PROJECT}} either -- accessor-only access doesn't include list, which is normal." >&2
            echo "  If '{{NAME}}' really is misnamed, ask someone with broader access to check." >&2
        else
            echo "  gcloud could not list secrets:" >&2
            sed 's/^/    /' "$err" >&2
            echo "  -> run 'gcloud auth login' and confirm access to {{GCP_PROJECT}}, then retry." >&2
        fi
        exit 0
    fi
    # Anchored, and only to the generated-suffix shape. A bare substring match
    # would happily resolve one credential's name to a different secret that
    # merely contains it, and write that value into .env without anyone noticing.
    matches=$(printf '%s\n' "$all" | grep -E "^{{NAME}}(-[A-Za-z0-9]+)?$" || true)
    count=$(printf '%s' "$matches" | grep -c . || true)
    if [ "$count" -eq 0 ]; then
        echo "  No secret matching '{{NAME}}'. Available secrets:" >&2
        printf '%s\n' "$all" | sed 's/^/    /' >&2
        exit 0
    fi
    if [ "$count" -gt 1 ]; then
        echo "  '{{NAME}}' is ambiguous -- $count secrets match:" >&2
        printf '%s\n' "$matches" | sed 's/^/    /' >&2
        echo "  Resolve by hand rather than guess; set the value in .env yourself." >&2
        exit 0
    fi
    name="$matches"
    value=$(gcloud secrets versions access latest --secret="$name" --project={{GCP_PROJECT}} 2>/dev/null || true)
    [ -n "$value" ] && echo "  Matched $name." >&2
    printf '%s' "$value"

# ── Development ──────────────────────────────────────

# google-adk reads GOOGLE_API_KEY straight from the process environment rather
# than through pydantic-settings, so .env has to be exported into the process.
# That's what `uv run --env-file` does here.
#
# Run the harness locally, with the MCP services mounted in-process
dev: _check-env
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "$(grep -s '^GOOGLE_API_KEY=' {{ENV_FILE}} | cut -d= -f2-)" ]; then
        echo "ERROR: GOOGLE_API_KEY is blank in {{ENV_FILE}} -- run 'just env' to fill it." >&2
        exit 1
    fi
    echo "Harness → http://localhost:{{PORT}}/healthz"
    echo "  data-mcp         → http://localhost:{{PORT}}/mcp/data/mcp"
    echo "  analysis-mcp     → http://localhost:{{PORT}}/mcp/analysis/mcp"
    echo "  presentation-mcp → http://localhost:{{PORT}}/mcp/presentation/mcp"
    echo "    (presentation-mcp fetches its release once at startup and mounts no tools"
    echo "     unless a dfo-apis instance with DFO_PRESENTATION_SURFACE=1 answers)"
    PORT={{PORT}} PRESENTATION_PUBLIC_URL=http://localhost:{{PORT}}/mcp/presentation \
        PRESENTATION_RELEASE_REQUIRED=false uv run --env-file {{ENV_FILE}} harness-server

# Serve the studio surface from deterministic scripted seams: no model, no
# upstream, no keys, sequential wire ids. Identical requests produce
# identical streams -- for frontend development and testing.
# The prompt picks the path: "comprehensive" -> plan pause, "ambiguous" ->
# elicitation, "nothing" -> no-data close, "fail" -> error frame,
# "canvas" -> answered from the canvas digest alone, no data fetch,
# "compare" -> direct fetch of a two-country frame as one two-series chart,
# "to the chart" / "to that chart" / "same chart" -> the same fetch, as a
# change to the newest card on the canvas instead of a new one,
# "misnamed" -> an approved plan whose analysis step refuses,
# anything else -> direct fetch + one card.
studio-scripted:
    @echo "Scripted studio → http://localhost:{{PORT}}/api/studio/turn (any Bearer token)"
    STUDIO_SCRIPTED=true SEED_FIXTURE_FRAMES=true PORT={{PORT}} \
        PRESENTATION_PUBLIC_URL=http://localhost:{{PORT}}/mcp/presentation \
        PRESENTATION_RELEASE_REQUIRED=false uv run harness-server

# Serve the studio surface against the real model and the real platform: a
# question is answered by the planner, not a script. Needs GOOGLE_API_KEY in
# {{ENV_FILE}}. Costs model spend on every turn.
# A plan whose steps include an analysis verb runs and narrates its answer,
# then cannot render a card: the platform refuses a frame_ref an analysis verb
# minted. The turn still reaches `done` with the figures in prose.
studio-live: _check-env
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "$(grep -s '^GOOGLE_API_KEY=' {{ENV_FILE}} | cut -d= -f2-)" ]; then
        echo "ERROR: GOOGLE_API_KEY is blank in {{ENV_FILE}} -- run 'just env' to fill it." >&2
        exit 1
    fi
    echo "Live studio → http://localhost:{{PORT}}/api/studio/turn (any Bearer token)"
    echo "  platform   → {{DFO_APIS_URL}}"
    echo "  startup takes 15-20s while presentation-mcp fetches its release"
    DFO_APIS_URL={{DFO_APIS_URL}} SEED_FIXTURE_FRAMES=true PORT={{PORT}} \
        PRESENTATION_PUBLIC_URL=http://localhost:{{PORT}}/mcp/presentation \
        uv run --env-file {{ENV_FILE}} harness-server

# Ask the harness one question from a terminal: POST /api/studio/turn,
# render every frame, and prompt through any plan or elicitation pause --
# no interrupt_id to copy by hand. Works against studio-scripted or
# studio-live (same contract, same port). Pass --yes to skip prompts.
# ARGS reaches the script as "$@" (positional-arguments, set above), never
# through {{ARGS}} text substitution -- a question containing shell
# metacharacters (';', quotes, ...) must stay inert, not run as a second
# command. Loads {{ENV_FILE}} when present, matching studio-live, so a real
# STUDIO_API_TOKEN there reaches this script instead of the CLI falling back
# to a token the live server would reject; studio-scripted needs no env
# file at all, so its absence stays fine.
ask *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -f "{{ENV_FILE}}" ]; then
        uv run --package harness --env-file {{ENV_FILE}} python scripts/ask.py "$@"
    else
        uv run --package harness python scripts/ask.py "$@"
    fi

# Ask the deployed dev harness ({{STUDIO_DEV_URL}}) instead of a local one --
# no --url to type or remember. `--url` is its own argv entry placed ahead
# of "$@", not text-substituted into it, so it can't be swallowed by
# `question`'s nargs="+" or reinterpreted as part of the question. Requires
# {{ENV_FILE}} (via _check-env) because the deployed server enforces
# STUDIO_API_TOKEN and rejects the CLI's placeholder with a 401.
ask-dev *ARGS: _check-env
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "$(grep -s '^STUDIO_API_TOKEN=' {{ENV_FILE}} | cut -d= -f2-)" ]; then
        echo "ERROR: STUDIO_API_TOKEN is blank in {{ENV_FILE}} -- run 'just env' to fill it; the deployed dev harness enforces it and rejects the CLI's placeholder token." >&2
        exit 1
    fi
    uv run --package harness --env-file {{ENV_FILE}} python scripts/ask.py --url {{STUDIO_DEV_URL}} "$@"

# Plain `uv sync` only syncs the (virtual) workspace root -- use this instead.
#
# Install every workspace member into the shared .venv
sync:
    uv sync --all-packages

# ── Evals ────────────────────────────────────────────
#
# Never run in CI: these drive the real agent through the real model, so they
# need a live credential. See evals/README.md.
#
# `google-adk[eval]`'s extras (pandas, tabulate, the judge-model plumbing)
# aren't any service's dependency, so `uv run --with` pulls them in
# ephemerally. The agent folders import `planner`, `translator` and `compiler`,
# which `just sync` installs into the shared .venv -- run that first if the
# import fails.

PLANNER_AGENT    := "evals/agents/harness_planner_v2"
PLANNER_SET      := "harness-planner-v2"
PAIR_AGENT       := "evals/agents/harness_pair_v1"
PAIR_SET         := "harness-pair-v1"
SKILLS_AGENT     := "evals/agents/harness_planner_skills_v1"
SKILLS_SET       := "harness-planner-skills-v1"
EVAL_ADK         := 'uv run --with "google-adk[eval]"'

# Port the ADK web UI listens on. `just dev` owns {{PORT}}, and the web UI
# has to reach those mounts, so the two cannot share one.
ADK_WEB_PORT     := "8000"

# --config_file_path is not optional. Omitted, the CLI silently substitutes
# ADK's hardcoded defaults instead of test_config.json's criteria.
#
# Score harness-planner-v2 from the CLI
eval-planner: _check-env
    {{EVAL_ADK}} adk eval {{PLANNER_AGENT}} {{PLANNER_SET}} \
        --config_file_path {{PLANNER_AGENT}}/test_config.json \
        --print_detailed_results

# Score harness-pair-v1 -- the planner with its translate tool -- from the CLI
eval-pair: _check-env
    {{EVAL_ADK}} adk eval {{PAIR_AGENT}} {{PAIR_SET}} \
        --config_file_path {{PAIR_AGENT}}/test_config.json \
        --print_detailed_results

# Score harness-planner-skills-v1 -- does the planner load the skill a question needs -- from the CLI
eval-planner-skills: _check-env
    {{EVAL_ADK}} adk eval {{SKILLS_AGENT}} {{SKILLS_SET}} \
        --config_file_path {{SKILLS_AGENT}}/test_config.json \
        --print_detailed_results

# The UI takes the agents dir, not one agent folder, and finds eval sets by
# scanning each agent folder for *.evalset.json.
#
# Its "Run evaluation" dialog cannot grade these eval sets: it offers only
# tool_trajectory_avg_score and response_match_score, sends no match_type (so
# trajectory matching falls back to EXACT), and cannot reach the rubric metrics.
# A healthy agent scores ~0 through it. Score with the `eval-*` recipes above.
#
# Browse cases, tool traces, and past run results in the ADK web UI
eval-web: _check-env
    @echo "Eval UI → http://127.0.0.1:8000 (pick an agent, then the Eval tab)"
    @echo "For reading runs. Score from the CLI -- not the Run evaluation dialog."
    {{EVAL_ADK}} adk web evals/agents

# Read a real turn's events in the ADK web UI: the whole system as one app,
# not the eval agents `eval-web` lists. Needs `just dev` running.
adk-web: _check-env
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "$(grep -s '^GOOGLE_API_KEY=' {{ENV_FILE}} | cut -d= -f2-)" ]; then
        echo "ERROR: GOOGLE_API_KEY is blank in {{ENV_FILE}} -- run 'just env' to fill it." >&2
        exit 1
    fi
    token="$(grep -s '^MCP_API_TOKEN=' {{ENV_FILE}} | cut -d= -f2-)"
    if [ -z "$token" ]; then
        echo "ERROR: MCP_API_TOKEN is blank in {{ENV_FILE}} -- run 'just env' to fill it." >&2
        echo "  The MCP mounts require the token 'just dev' resolved; without the same value here, every tool call 401s." >&2
        exit 1
    fi
    # An empty JSON body is not a valid MCP message, so a healthy mount that
    # accepts the bearer answers 400 with a JSON-RPC protocol error (and mints
    # no MCP session). A missing or wrong bearer is refused earlier, at the
    # auth middleware, with 401. Nothing listening is 000. Every other status
    # (404 from a wrong server or a missing mount, 405, 5xx from a broken
    # harness) is refused rather than passed through to fail later.
    code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:{{PORT}}/mcp/data/mcp" \
        -H "Authorization: Bearer $token" -H 'Content-Type: application/json' \
        -H 'Accept: application/json, text/event-stream' -d '{}' || true)"
    if [ "$code" = "000" ]; then
        echo "ERROR: nothing is serving the MCP mounts on port {{PORT}} -- run 'just dev' (or 'just studio-live') in another terminal first." >&2
        exit 1
    fi
    if [ "$code" = "401" ]; then
        echo "ERROR: the harness on port {{PORT}} rejects MCP_API_TOKEN from {{ENV_FILE}} -- restart 'just dev' after setting it, since a server with none minted a token only it knows." >&2
        exit 1
    fi
    if [ "$code" != "400" ]; then
        echo "ERROR: POST /mcp/data/mcp on port {{PORT}} answered HTTP $code, not the 400 protocol error a healthy MCP mount returns -- check that 'just dev' is what serves port {{PORT}} and that its log shows the mounts started." >&2
        exit 1
    fi
    echo "ADK web → http://127.0.0.1:{{ADK_WEB_PORT}} (app: dfo_analyst_graph)"
    echo "  MCP mounts → http://127.0.0.1:{{PORT}}/mcp/{data,analysis,presentation}/mcp"
    echo "  Startup takes 15-20s: translation reads both MCP servers' schemas."
    PORT={{PORT}} uv run --env-file {{ENV_FILE}} --with "google-adk" adk web evals/agents \
        --port {{ADK_WEB_PORT}} --session_service_uri memory://

# harness-studio-v1's contract layer: deterministic, in-process, no key.
# (CI runs the same battery via services/harness/tests/test_studio_conformance.py.)
eval-studio-contract:
    uv run --package harness python evals/studio/run.py --mode scripted

# The orchestrator engine's model-free contract. The real graph and planner seam
# run with scripted models, and every model request is recorded and checked.
# Cases marked xfail hold a known gap. See evals/studio/README.md.
eval-studio-orchestrator:
    uv run --package harness python evals/studio/run.py --mode orchestrator

# harness-studio-v1 in full: live agent + rubric judge. Start the harness
# first (and the datasets stub if the hosted mint is down); needs the model
# key. See evals/studio/README.md.
eval-studio: _check-env
    uv run --env-file {{ENV_FILE}} --package harness python evals/studio/run.py \
        --mode live --url http://localhost:{{PORT}} --judge

# The orchestrator engine's live cases with the rubric judge. Start the harness
# with `STUDIO_ENGINE=orchestrator just studio-live` first; needs the model key.
eval-studio-orchestrator-live: _check-env
    uv run --env-file {{ENV_FILE}} --package harness python evals/studio/run.py \
        --mode orchestrator-live --url http://localhost:{{PORT}} --judge

# ── Quality ──────────────────────────────────────────

# Lint (no fixes) + format check, across the whole workspace
lint:
    uv run ruff check .
    uv run ruff format --check .

# Lint and format, applying fixes
fix:
    uv run ruff check --fix .
    uv run ruff format .

# Type check every workspace member with ty
# (services/harness/src covers both harness and planner -- see
# services/harness/pyproject.toml's tool.uv.build-backend.module-name)
typecheck:
    uv run ty check packages/common/src
    uv run ty check services/harness/src
    uv run ty check services/data-mcp/src
    uv run ty check services/analysis-mcp/src
    uv run ty check services/presentation-mcp/src

# Root pyproject.toml's [tool.pytest.ini_options] lists each member's testpaths
# and sets --import-mode=importlib so same-named test modules across
# packages/services (e.g. test_settings.py) don't collide.
#
# Run every workspace member's test suite in one invocation from the root
test:
    uv run pytest

# CI's third job builds the image. `docker-build` runs it locally.
#
# Everything CI's lint and test jobs run, in CI's order. Green predicts green
check: lint typecheck test

# Build context is the repo root, so the Dockerfile's paths resolve from there.
#
# Build the Cloud Run image the way CI's docker job does
docker-build:
    docker buildx build -f deploy/Dockerfile.ai -t dfo-ai:local .

# --- Per-package test shortcuts ---
#
# The workspace root's pyproject.toml carries the only [tool.pytest.ini_options]
# table in the tree, so pytest's rootdir is always the workspace root, whatever
# path or package these point at (including a bare `pytest services/X/tests`
# run directly). That keeps the workspace conftest.py, the one that isolates the
# suite from a local .env, in scope for every invocation. Coverage flags live
# here rather than in each member's pyproject.toml so that table stays absent.
# A `[tool.pytest.ini_options]` table there would shadow the root one for any
# pytest invocation that resolves to it, dropping the workspace conftest.py
# silently. test_single_workspace_pytest_config.py guards against that.

test-common:
    uv run --package common pytest packages/common -v \
        --cov=packages/common/src/common --cov-report=term --cov-report=xml

test-harness:
    uv run --package harness pytest services/harness/tests -v \
        --cov=services/harness/src/harness --cov=services/harness/src/planner \
        --cov=services/harness/src/compiler --cov=services/harness/src/translator \
        --cov=services/harness/src/executor --cov=services/harness/src/orchestrator \
        --cov-report=term --cov-report=xml

test-data-mcp:
    uv run --package data_mcp pytest services/data-mcp/tests -v \
        --cov=services/data-mcp/src/data_mcp --cov-report=term --cov-report=xml

test-analysis-mcp:
    uv run --package analysis_mcp pytest services/analysis-mcp/tests -v \
        --cov=services/analysis-mcp/src/analysis_mcp --cov-report=term --cov-report=xml

test-presentation-mcp:
    uv run --package presentation_mcp pytest services/presentation-mcp/tests -v \
        --cov=services/presentation-mcp/src/presentation_mcp --cov-report=term --cov-report=xml

# Peak memory and wall time of the frame path at 50k, 60k and 250k rows, and the frame row
# budget they imply at `--memory-mib` (default 2048). Pass `--out FILE` to keep the JSON,
# `--rows N` or `--steps a,b` to narrow it (see scripts/bench_frames.py)
bench-frames *args:
    uv run python scripts/bench_frames.py {{args}}

# Re-measure the digest corpus and rewrite the committed baseline
digest-baseline:
    DIGEST_BASELINE_UPDATE=1 uv run pytest services/data-mcp/tests/test_digest_budget.py -q

# Re-measure analysis-mcp's aggregate digest corpus and rewrite its committed baseline
analysis-digest-baseline:
    DIGEST_BASELINE_UPDATE=1 uv run pytest services/analysis-mcp/tests/test_digest_budget.py -q

# Re-measure what the planner model is handed and rewrite the committed
# composition baseline. Run after any deliberate change to the planner's
# instruction text, tool surface or history handling. The
# diff of services/harness/tests/context_baseline.json is the review artifact.
context-baseline:
    CONTEXT_BASELINE_UPDATE=1 uv run pytest services/harness/tests/test_context_budget.py --confcutdir=. -q

# ── Diagnostics ──────────────────────────────────────

# Check the local environment: tooling, gcloud auth, env file, workspace, hooks
doctor:
    #!/usr/bin/env bash
    echo "Tooling:"
    for tool in uv gcloud gh git; do
        if command -v "$tool" >/dev/null 2>&1; then
            echo "  ok    $tool"
        else
            echo "  MISS  $tool"
        fi
    done
    echo ""
    echo "GCP ({{GCP_PROJECT}}):"
    account=$(gcloud config get-value account 2>/dev/null || true)
    if [ -n "$account" ] && [ "$account" != "(unset)" ]; then
        echo "  ok    authenticated as $account"
    else
        echo "  MISS  not authenticated -- run 'gcloud auth login'"
    fi
    if [ -f "$HOME/.config/gcloud/application_default_credentials.json" ]; then
        echo "  ok    application default credentials"
    else
        echo "  MISS  no ADC -- run 'gcloud auth application-default login'"
    fi
    # secretAccessor doesn't grant secrets.list, so gating on that call would
    # flag a correctly-provisioned developer. Probe specific secrets instead.
    check_secret() {
        local name="$1" required="$2"
        if gcloud secrets versions access latest --secret="$name" --project={{GCP_PROJECT}} >/dev/null 2>&1; then
            echo "  ok    $name"
        elif [ "$required" = required ]; then
            echo "  MISS  $name -- not readable (missing grant, or 'just env' needs it)"
        else
            echo "  off   $name -- not readable (tracing stays off; grant is optional)"
        fi
    }
    check_secret dfo-dev-gemini-api-key      required
    check_secret dfo-dev-langfuse-public-key optional
    check_secret dfo-dev-langfuse-secret-key optional
    echo ""
    echo "Env file ({{ENV_FILE}}):"
    if [ ! -f "{{ENV_FILE}}" ]; then
        echo "  MISS  not found -- run 'just env'"
    else
        # Required to answer a question at all; the Langfuse pair only gates tracing.
        for key in GOOGLE_API_KEY DFO_APIS_URL MODEL_NAME; do
            if [ -n "$(grep -s "^${key}=" {{ENV_FILE}} | head -1 | cut -d= -f2-)" ]; then
                echo "  ok    $key"
            else
                echo "  MISS  $key -- run 'just env'"
            fi
        done
        for key in LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY LANGFUSE_BASE_URL; do
            if [ -n "$(grep -s "^${key}=" {{ENV_FILE}} | head -1 | cut -d= -f2-)" ]; then
                echo "  ok    $key"
            else
                echo "  off   $key (tracing disabled)"
            fi
        done
        # A key added to the sample after a developer's .env was written would
        # otherwise sit unnoticed until the code that reads it fails.
        missing=$(comm -23 \
            <(grep -oE '^[A-Z_0-9]+=' {{ENV_SAMPLE}} | sort -u) \
            <(grep -oE '^[A-Z_0-9]+=' {{ENV_FILE}} | sort -u))
        if [ -n "$missing" ]; then
            echo "  NOTE  keys in {{ENV_SAMPLE}} but not in {{ENV_FILE}}:"
            printf '%s\n' "$missing" | sed 's/=$//; s/^/          /'
            echo "        run 'just env' to add them"
        fi
    fi
    echo ""
    echo "Workspace:"
    if [ -d .venv ]; then
        echo "  ok    .venv present"
    else
        echo "  MISS  no .venv -- run 'just sync'"
    fi
    if [ -f .git/hooks/pre-commit ]; then
        echo "  ok    pre-commit hooks installed"
    else
        echo "  MISS  hooks not installed -- run 'uv run pre-commit install'"
    fi

# ── Internal ─────────────────────────────────────────

_check-env:
    #!/usr/bin/env bash
    if [ ! -f "{{ENV_FILE}}" ]; then
        echo "Error: {{ENV_FILE}} not found."
        echo "Run 'just setup' or 'just env' first."
        exit 1
    fi

_check-prerequisites:
    #!/usr/bin/env bash
    set -euo pipefail
    missing=()
    for tool in uv gcloud; do
        command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "Error: missing prerequisites: ${missing[*]}"
        echo "  uv:     https://docs.astral.sh/uv/getting-started/installation/"
        echo "  gcloud: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    account=$(gcloud config get-value account 2>/dev/null || true)
    if [ -z "$account" ] || [ "$account" = "(unset)" ]; then
        echo "Error: gcloud is not authenticated. Run 'gcloud auth login'."
        exit 1
    fi
    echo "Prerequisites ok (gcloud: $account)."
