# MAGs-CodeDev

**M**ulti-**A**gent **G**raph **S**ystem for **Code** **Dev**elopment.

MAGs-CodeDev is an autonomous, multi-agent AI software engineer built for the command line. You give it a **manifest** of modules (with descriptions and dependencies), and it builds them as a **DAG** — running each module through an agent loop (code → test → diagnose → review → iterate) in an isolated git worktree + container, then merges to `main` on success.

```
  ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐
  │ Coder   │──▶│ Tester  │──▶│ LogCheck │──▶│ Reviewer │
  │ (write  │   │ (write  │   │ (triage  │   │ (multi-  │
  │  code)  │   │  tests) │   │  errors) │   │  LLM)    │
  └────┬────┘   └────┬────┘   └─────┬────┘   └─────┬────┘
       │              │              │               │
       │    ┌─▶ run tests ─▶ results ◀─┘             │
       │    │   └─▶ run lints ─▶ results ◀───────────┘
       │    │      │
       │    │  ┌───┴──────────────────────────────────┐
       │    │  │ Convergence check (SHA-256 hashes)   │
       │    │  │  changed → review                     │
       │    │  │  unchanged → fail (no progress)       │
       │    │  └──────────────────────────────────────┘
       │    │         │
       │    │    ┌─────┴────────────────────────────────┐
       │    │    │ Review: majority LGTM → merge        │
       │    │    │  actionable comments → back to coder │
       │    └────┴──────────────────────────────────────┘
       │         log_checker routes: fix_source → coder
       └────────                   fix_tests  → tester
```

Each module runs in its own **git worktree** and **container**, with independent modules building concurrently in DAG waves. Failed modules don't block the rest.

---

## Installation

### Standalone CLI

```bash
# Clone and install
git clone <repo>
cd MAGs-CodeDev
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Verify
mags-codedev --help
```

### OMP Extension (streaming + structured)

For integration with [Oh My Pi](https://omp.sh) — streaming JSONL progress, structured results, session auto-detect:

```bash
# Prerequisites: OMP installed
bun install -g @oh-my-pi/pi-coding-agent

# Install the CLI (same as standalone)
cd MAGs-CodeDev && pip install -e .

# Install the extension
omp plugin link --scope=project ./mags-codedev-extension

# Verify
omp -p '/extensions'   # should show mags-codedev-extension (7 tools)
```

### Container Sandbox (full isolation)

Moved to the [omp-sandbox](https://github.com/TimothyStephens/omp-sandbox) repo (minimal OMP container + version-controlled provisioning of `~/.omp`):

```bash
cd omp-sandbox
bash Containerfile_build   # builds localhost/omp-sandbox:latest
./omp-sandbox install      # provisions ~/.omp from manifest.json
./omp-workspace            # tmux + podman session for a project
```

### Dependencies

| Category | Packages |
|---|---|
| CLI | `typer`, `rich` |
| LLM | `langgraph`, `langchain-core`, `langchain-openai`, `langchain-google-genai`, `langchain-anthropic`, `openai`, `anthropic`, `google-generativeai`, `google-genai` |
| Git | `gitpython` |
| Config | `pyyaml`, `pydantic` |
| Resilience | `tenacity`, `httpx` |

### Container Runtime

Auto-detected in priority order: **podman > docker > apptainer > singularity > local**. Set `test_runner: "auto"` (default) or pin a specific runtime. The local runner installs the project's `requirements.txt` + test toolchain before running tests.

---

## Quick Start

### 1. Initialize

```bash
mags-codedev init
```

Creates `.mags-codedev/` (config, cache.db, logs, worktrees), `manifest.json`, `AGENT.md`, `.gitignore`, and a git repo if needed. In interactive mode, the AI Architect helps design the project structure.

| Flag | Description |
|---|---|
| `--non-interactive` | Skip editor + AI Architect |
| `-m PATH` | Custom manifest path (default: `<base_dir>/manifest.json`) |
| `-c PATH` | Custom config path |

### 2. Configure API Keys

Edit `.mags-codedev/config.yaml`. Keys can also be set via environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, etc.) — env vars take priority.

### 3. Edit the Manifest

```json
[
  {
    "location": "src/core/types.py",
    "description": "Data classes and type definitions for the domain model.",
    "dependencies": []
  },
  {
    "location": "src/core/validator.py",
    "description": "Input validation using pydantic models from types.",
    "dependencies": ["src/core/types.py"]
  }
]
```

- `location` — relative path to the module file
- `description` — instructions given to the coder agent
- `dependencies` — array of `location` values this module imports from

### 4. Build

```bash
mags-codedev build
```

Builds modules in DAG dependency order (topological waves). Each module runs the agent loop in an isolated worktree + container. On success, commits to a `feature/<location>` branch and merges to `main`. Failed modules are reported with their log file path — independent modules in other waves still run.

| Flag | Description |
|---|---|
| `--module <location>` | Build only this module (forces a rebuild even if already built) |
| `--force-fresh` | Delete all worktrees/branches, rebuild everything |
| `--skip-validation` | Skip the pre-build LLM connection check |
| `--json` | Emit JSONL status events to stdout (for OMP / CI integration) |
| `-v` / `-vv` | Debug (full LLM chat + file contents) / trace |
| `-m PATH` | Custom manifest path |
| `-c PATH` | Custom config path |

---

## CLI Commands

### `mags-codedev init`

Initialize a workspace.

```bash
mags-codedev init
mags-codedev init --non-interactive
```

### `mags-codedev build`

Build all pending modules (or a single module with `--module`).

```bash
mags-codedev build
mags-codedev build --module src/pricing.py    # rerun one task
mags-codedev build --force-fresh              # rebuild everything
mags-codedev build --json                     # JSONL output for OMP/CI
mags-codedev build -v                         # debug (full LLM chat in logs)
```

### `mags-codedev test`

Run project tests in the configured container environment.

```bash
mags-codedev test
mags-codedev test -v
```

### `mags-codedev debug <ERROR_MSG>`

Pass an error trace or log file path to the LLM for automatic fixing. Auto-detects the module from the log file hash.

```bash
mags-codedev debug "ImportError: cannot import name 'X' from 'Y'"
mags-codedev debug .mags-codedev/logs/<hash>.log
mags-codedev debug "Bug in pricing" --mod src/pricing.py
```

With `--mod` (or a log path whose hash auto-detects the module), `debug` re-runs the module's full fix loop automatically (cmd_debug.py:173-178). Without a resolvable module it only analyzes the trace — then rerun via `mags-codedev build --module <location>`.

### `mags-codedev chat`

Interactive chat with the LLM about the codebase (can read/write files).

```bash
mags-codedev chat
```

### `mags-codedev tokens`

Display token usage by role and model.

```bash
mags-codedev tokens
```

### `mags-codedev list-models`

List available models from configured providers.

```bash
mags-codedev list-models
```

### `mags-codedev clean`

Remove all generated artifacts.

```bash
mags-codedev clean
mags-codedev clean --force    # skip confirmation
```

---

## Configuration

Config lives at `.mags-codedev/config.yaml`. Override with `-c`.

```yaml
api_keys:
  openai: "sk-..."
  anthropic: "sk-ant-..."
  gemini: "AIza..."
  ollama: "ollama-api-key-..."   # for local providers

models:
  # Interactive commands (init, chat, debug)
  interactive_commands:
    chat:
      provider: "openai"
      model: "gpt-4o"

  # Build workflow agents
  build_workflow:
    coder:
      provider: "openai"
      model: "gpt-4o"
    tester:
      provider: "anthropic"
      model: "claude-3-5-sonnet-20240620"
    log_checker:
      provider: "google"
      model: "gemini-2.5-pro"
    reviewers:                      # multiple reviewers run concurrently
      - provider: "openai"
        model: "gpt-4o"
      - provider: "anthropic"
        model: "claude-3-5-sonnet-20240620"
      # Local model via OpenAI-compatible endpoint (Ollama, vLLM, LM Studio)
      - provider: "local"
        model: "llama3:70b"
        base_url: "http://localhost:11434/v1"

settings:
  language: "python"

  # Container runtime: auto | podman | docker | apptainer | singularity | local
  test_runner: "auto"

  # Container image names (auto-built if missing)
  docker_test_image: "mags-dev-env:latest"
  apptainer_test_image: "mags-dev-env.sif"

  # Parallelism and budgets
  max_parallel_modules: 4
  max_test_fix_iterations: 5     # max test/lint fix cycles before aborting
  max_review_rounds: 3          # max review revision rounds before aborting
  timeout_per_module_mins: 15  # max minutes for each container test/lint run

  # Artifact directory (auto-gitignored)
  base_dir: ".mags-codedev"

  # Logging: info | debug | trace
  log_level: "info"
```

### Environment Variables

API keys and model config can be set via env vars (priority over YAML):

| Env Var | Maps to |
|---|---|
| `OPENAI_API_KEY` | `api_keys.openai` |
| `ANTHROPIC_API_KEY` | `api_keys.anthropic` |
| `GOOGLE_API_KEY` | `api_keys.gemini` |
| `OLLAMA_API_KEY` | `api_keys.ollama` |
| `MAGS_MODEL` | Override model for all roles |
| `MAGS_PROVIDER` | Override provider for all roles |
| `MAGS_MODEL_CODER` | Override model for the coder role only |

### Local Providers

For local models (Ollama, vLLM, LM Studio), use `provider: "local"` (or legacy `"custom_openai"`):

```yaml
reviewers:
  - provider: "local"
    model: "Qwen2.5-72B"
    base_url: "http://localhost:8000/v1"
```

Ollama can also be configured directly:

```yaml
reviewers:
  - provider: "ollama"
    model: "llama3.1"
    base_url: "http://localhost:11434"   # optional, defaults to this
    num_ctx: 8192                        # optional context window
```

`ollama` provider prerequisite: `pip install langchain-ollama` (not bundled with the standard install; config_parser.py:253 raises a clear ImportError without it).

---

## Architecture

### Agent Loop (per module)

Each module runs its own LangGraph:

```
session_start → coder → tester → run_tests → run_linters → log_checker
     ↑                                                         │
     │         ┌───────────────────────────────────────────────┘
     │         ▼
     │    check_convergence → multi_llm_review → (approved? → session_end)
     │         │                       │
     │    (unchanged → fail)      (revise → back to coder)
     │
     └── log_checker routes: fix_source → coder, fix_tests → tester
```

**Nodes:**

1. **Coder** — LLM writes module code from the spec + dependency source. On fix cycles, receives error summaries + reviewer comments + existing code.
2. **Tester** — LLM writes pytest tests. On fix cycles, receives the diagnosis + broken tests.
3. **run_tests / run_linters** — Executes pytest/flake8/mypy in an isolated container mounted on the module's git worktree.
4. **LogChecker** — LLM analyzes test/lint output, outputs JSON `{location, summary}`. Routes to coder (source error) or tester (test error).
5. **check_convergence** — SHA-256 of code + tests. If unchanged from previous iteration, the module fails (prevents infinite loops without progress).
6. **multi_llm_review** — N reviewers run concurrently via `asyncio.gather`. Approval requires a strict majority of all configured reviewers to reply `LGTM`. Skipped reviewers (API failure) are neutral — never counted as a vote.

### LLM Reliability

- **Transient errors** (rate limits, 503s, server disconnects) are retried with exponential backoff (5 attempts, 4–60s waits).
- **Hard failures** (auth, quota) fail the task with a real, logged error — no silent stub fallbacks. A failed module's last good code is saved to the artifact DB for inspection.
- **Reviewer quorum** — a skipped reviewer is neutral, not an approval. Approval requires `len(approvals) * 2 > len(reviewers)`. A partial API outage can never grant a 1-of-N approval.

### DAG Parallelism

Modules with satisfied dependencies build concurrently in waves:

```
Wave 1:  [types.py] [config.py] [constants.py]     ← 3 parallel
Wave 2:  [validator.py] [utils.py]                  ← 2 parallel
Wave 3:  [engine.py]                                ← 1
Wave 4:  [pipeline.py] [cli.py]                     ← 2 parallel
```

Each module gets its own:
- **Git worktree** (isolated from other modules)
- **LangGraph** (isolated state)
- **Container** (isolated test environment)
- **Log file** (`.mags-codedev/logs/<hash>.log`)

If a module fails, its dependents are marked blocked (not built), but independent modules in other waves still run.

### Spec-Aware Rebuilds

Completed modules are cached by a location hash (for log/worktree continuity) **and** a spec content hash (description + dependencies). Editing a module's `description` or `dependencies` in the manifest automatically invalidates it on the next `build` — no `--force-fresh` needed. Use `--module <location>` to rerun a single task.

### Convergence Detection

A module is marked as failed when the code or test hash is **identical** to the previous iteration — meaning the agent loop isn't making progress. This catches cases where the coder keeps regenerating the same broken code. The iteration budget (`max_test_fix_iterations`) and review budget (`max_review_rounds`) are separate.

### Token Tracking

All LLM calls are tracked via `TokenLoggingCallbackHandler` (persisted to SQLite) and `TokenCounter` (in-memory for the live status tree). View with `mags-codedev tokens`.

### Conversation Logging

Module logs (`.mags-codedev/logs/<hash>.log`) contain the full LLM conversation:

- **INFO (default)** — concise exchange summary: task narrative, payload line-counts (code/tests/deps elided to counts, not dumped), and inter-agent diagnostics (error summaries, reviewer comments). This is the "chat between LLMs" without file contents.
- **DEBUG (`-v`)** — full prompt + full response, including embedded file contents.

This lets you inspect a failed task's LLM conversation without wading through large code dumps — unless you want them.

---

## JSONL Output (`--json`)

When `mags-codedev build --json` is used, the CLI emits one JSON object per line (JSONL) to stdout instead of the Rich Live tree. This is consumed by the OMP extension for streaming progress, but can also be used by CI pipelines or `jq`.

```jsonl
{"ts":"...","event":"build_start","manifest":"manifest.json","total_modules":5,"already_built":2}
{"ts":"...","event":"module_start","location":"src/pricing.py","hash":"a1b2...","session":1}
{"ts":"...","event":"module_step","location":"src/pricing.py","step":"coder","iteration":1}
{"ts":"...","event":"module_end","location":"src/pricing.py","status":"Success: Merged to Main","iterations":2,"log_file":".mags/logs/a1b2.log","tokens_in":4200,"tokens_out":1800}
{"ts":"...","event":"build_end","total_modules":5,"succeeded":4,"failed":1,"blocked":0,"tokens_in":45000,"tokens_out":12000}
```

| Event | Key fields |
|---|---|
| `build_start` | `manifest`, `total_modules`, `already_built` |
| `module_start` | `location`, `hash`, `session` |
| `module_step` | `location`, `step`, `iteration` |
| `module_tokens` | `location`, `tokens_in`, `tokens_out` |
| `module_end` | `location`, `status`, `iterations`, `log_file`, `tokens_in`, `tokens_out` |
| `wave_end` | `succeeded`, `failed` |
| `build_end` | `total_modules`, `succeeded`, `failed`, `blocked`, `tokens_in`, `tokens_out` |

---

## OMP Extension

The `mags-codedev-extension/` directory contains an OMP extension that wraps the CLI as structured tools with streaming JSONL progress.

### Tools

| Tool | CLI equivalent | Key params |
|---|---|---|
| `mags_init` | `mags-codedev init` | `interactive` |
| `mags_build` | `mags-codedev build` | `module`, `force_fresh`, `json`, `verbose` |
| `mags_test` | `mags-codedev test` | `verbose` |
| `mags_debug` | `mags-codedev debug` | `error_msg`, `module_location` |
| `mags_tokens` | `mags-codedev tokens` | — |
| `mags_list_models` | `mags-codedev list-models` | `config_path` |
| `mags_clean` | `mags-codedev clean` | `force` |

When `mags_build` is called from OMP, it defaults to `json=true` — the CLI runs with `--json`, the extension parses JSONL events, and streams `module_step` progress to the OMP TUI via `onUpdate`. The final `build_end` event is returned as structured `details` so the agent can act on failures programmatically.

A `session_start` hook auto-detects a MAGs-CodeDev workspace (`.mags-codedev/manifest.json`) and notifies the user.

---

## Test Environment

### Running the MAGs-CodeDev test suite

```bash
python3 -m pytest tests/ -v
python3 -m pytest tests/ --cov=mags_codedev --cov-report=term-missing
```

### Container Test Environment

Module tests run in isolated containers. The framework:
1. Detects the container runtime (podman > docker > apptainer > singularity)
2. Builds an image from the project's `requirements.txt` + language toolchain
3. Runs pytest/flake8/mypy inside the container
4. Mounts the worktree so the container sees the module code

To test locally without containers, set `test_runner: "local"` in config.

---

## Manifest Format

A JSON array of module objects. Order does not matter — the build system computes topological order.

```json
[
  {
    "location": "src/module.py",
    "description": "What this module does, what it should contain, conventions to follow.",
    "dependencies": ["src/other_module.py"]
  }
]
```

### Rules

- `location` must be a relative path to a file
- `dependencies` must reference `location` values of other modules in the manifest
- Cycles and missing dependencies are detected during scheduling — independent modules may build and merge first, then the build exits with an error listing the stuck modules (cmd_build.py:724-741).
- Modules with no dependencies build in the first wave
- Editing a module's `description` or `dependencies` invalidates that module on the next `build` (spec-aware rebuild). The location-based hash is preserved for log/worktree continuity.

---

## Troubleshooting

### "Module already built" but code is stale

Spec edits auto-invalidate. To force a full rebuild:

```bash
mags-codedev build --force-fresh
```

To rerun a single module:

```bash
mags-codedev build --module src/pricing.py
```

### Module failed — how to inspect and fix

```bash
# Find the log file (printed in the build output, or in .mags-codedev/logs/)
# The log has the full LLM conversation (file contents elided at INFO level)

# Debug with the log (auto-detects the module)
mags-codedev debug .mags-codedev/logs/<hash>.log

# Or pass raw error text + module location
mags-codedev debug "AssertionError in pricing calculation" --mod src/pricing.py

# After fixing, rerun the task
mags-codedev build --module src/pricing.py
```

### Container build fails

```bash
podman info   # or: docker info
# Check the image build log — the Dockerfile is auto-generated from requirements.txt
```

### LLM API errors

```bash
mags-codedev list-models    # verify provider/model combinations
```

### Module stuck in infinite loop

The convergence check fails a module when code/tests stop changing between iterations. If you hit `max_test_fix_iterations` or `max_review_rounds`, check the module log with `mags-codedev debug`, fix the issue, and rerun with `--module`.

### Enable full LLM chat in logs

```bash
mags-codedev build -v        # debug: full prompts + responses + file contents
mags-codedev build -vv       # trace: token usage per call
```

---

## License

MIT
