# MAGs-CodeDev

Multi-agent LLM-powered software development workflow using [LangGraph](https://langchain-ai.github.io/langgraph/). Each module in a project is built by a dedicated agent graph that codes, tests, reviews, and iterates until convergence.

```
  ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐
  │ Coder   │──▶│ Tester  │──▶│ LogCheck │──▶│ Reviewer │
  │ (write) │   │ (write  │   │ (triage) │   │ (review) │
  │  code)  │   │  tests) │   │          │   │  code)   │
  └────┬────┘   └────┬────┘   └─────┬────┘   └─────┬────┘
       │              │              │               │
       └──▶ run tests ──▶ results ◀─┘               │
           └──▶ run lints ──▶ results ◀─────────────┘
              │
         ┌────┴─────────────────────────────────────┐
         │  Converged? ──no──▶ route back to Coder   │
         │  Converged? ──yes──▶ commit, mark done    │
         └───────────────────────────────────────────┘
```

Each module runs in its own **git worktree** and **container runtime**, with parallel modules building concurrently.

---

## Installation

### Option A: Editable install (recommended for development)

Uses `pyproject.toml` for dependency resolution and installs the package in editable mode.

```bash
# Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install in development mode
pip install -e .

# Verify installation
mags-codedev --help
```

### Option B: Requirements file

Install directly from `requirements.txt` (no editable mode).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Option C: Conda environment

```bash
conda create -n mags-codedev python=3.11 -y
conda activate mags-codedev
pip install -e .
```

### Option D: Mamba environment

```bash
mamba create -n mags-codedev python=3.11 -y
mamba activate mags-codedev
pip install -e .
```

> **Note:** MAGs-CodeDev depends on LangChain/LangGraph packages which are pure Python and not available as conda-forge packages. Use `pip install` inside the conda/mamba environment for these.

### Dependencies

| Category | Packages |
|---|---|
| CLI | `typer`, `rich` |
| LLM | `langgraph`, `langchain-core`, `langchain-openai`, `langchain-google-genai`, `langchain-anthropic`, `openai`, `anthropic`, `google-generativeai`, `google-genai` |
| Git | `gitpython` |
| Container | `docker` |
| Config | `pyyaml`, `pydantic` |
| Resilience | `tenacity`, `httpx` |

### Container Runtime

The framework auto-detects the best available container runtime:

1. **Podman** (preferred)
2. **Docker**
3. **Apptainer** (or Singularity — accepted as alias)
4. **Singularity**
5. **Local** (falls back to host environment)

Set `test_runner: "auto"` in config (default) for auto-detection, or pin a specific runtime.

---

## Quick Start

### 1. Initialize

```bash
mags-codedev init
```

This creates:
- `.mags-codedev/config.yaml` — copied from `~/.omp/agent/mags-codedev.yaml` (if it exists), or from the package template
- `.mags-codedev/cache.db` — SQLite database tracking build state
- `.mags-codedev/logs/` — per-module log files
- `.mags-codedev/worktrees/` — git worktrees for each module
- `.mags-codedev/containers/` — container image definitions
- `manifest.json` — module definitions (or AI Architect interactive session)
- `AGENT.md` — coding conventions for the LLM agents
- `.gitignore` — auto-configured to ignore `.mags-codedev/`
- Git repo (initialized if one doesn't exist)

**Options:**
| Flag | Description |
|---|---|
| `--non-interactive` | Skip editor and AI Architect mode |
| `--offline` | Skip LLM calls entirely |
| `-m PATH` | Custom manifest path (default: `manifest.json`) |
| `-c PATH` | Custom config path |

### 2. Configure API Keys

Edit `.mags-codedev/config.yaml`:

```yaml
api_keys:
  openai: "sk-..."
  anthropic: "sk-ant-..."
  gemini: "AIza..."
```

### 3. Edit the Manifest

`manifest.json` is a JSON array of module specifications:

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

The build system:
1. Loads the manifest and validates dependencies (no cycles)
2. Creates a dependency graph and computes build order (topological sort)
3. For each module, spawns a **parallel LangGraph** with:
   - **Coder** — writes module code based on description + dependency source
   - **Tester** — writes and runs tests in an isolated container
   - **LogChecker** — analyzes test/lint output, determines if failure is in code or tests
   - **Reviewers** — multiple LLM agents review code concurrently
4. Routes back to Coder or Tester if errors are found (up to `max_test_fix_iterations`)
5. Commits to the module's git worktree branch on success
6. Prints a live status tree with token usage

**Options:**
| Flag | Description |
|---|---|
| `--force-fresh` | Delete worktrees/branches, rebuild everything from scratch |
| `--skip-validation` | Skip the pre-build LLM connection check |
| `--offline` | Use stub LLMs (no API calls) |
| `-v` / `-vv` | Debug / trace verbosity |
| `-m PATH` | Custom manifest path |
| `-c PATH` | Custom config path |

---

## CLI Commands

### `mags-codedev init`

Initialize a new workspace.

```bash
mags-codedev init
mags-codedev init --non-interactive --offline
mags-codedev init -m custom_manifest.json -c /path/to/config.yaml
```

| Flag | Default | Description |
|---|---|---|
| `-m, --manifest` | `manifest.json` | Path for the manifest file |
| `-c, --config` | auto | Path to config.yaml |
| `--interactive/--non-interactive` | interactive | Open editor + AI Architect |
| `--offline` | false | Skip LLM API calls |

### `mags-codedev build`

Build all pending modules using parallel LangGraph agents.

```bash
mags-codedev build
mags-codedev build --force-fresh
mags-codedev build -vv                    # trace-level logging
```

| Flag | Default | Description |
|---|---|---|
| `-m, --manifest` | `manifest.json` | Manifest path |
| `-c, --config` | auto | Config path |
| `--force-fresh` | false | Rebuild everything, ignore DB cache |
| `--skip-validation` | false | Skip pre-build connection check |
| `--offline` | false | Stub LLMs |
| `-v` (count) | 0 | Verbosity: 0=info, 1=debug, 2=trace |

### `mags-codedev test`

Run project tests in the configured container environment.

```bash
mags-codedev test
mags-codedev test -v                      # debug logging
```

| Flag | Default | Description |
|---|---|---|
| `-c, --config` | auto | Config path |
| `--offline` | false | Stub LLMs |
| `-v` (count) | 0 | Verbosity |

### `mags-codedev debug <ERROR_MSG>`

Pass an error trace or bug description to the LLM for automatic fixing.

```bash
mags-codedev debug "ImportError: cannot import name 'X' from 'Y'"
mags-codedev debug /path/to/error.log
mags-codedev debug "Bug in pricing calculation" --mod src/pricing.py
```

| Argument | Description |
|---|---|
| `error_msg` | Error text or path to log file |

| Flag | Default | Description |
|---|---|---|
| `--mod, --module` | auto | Target module location |
| `-m, --manifest` | `manifest.json` | Manifest path |
| `-c, --config` | auto | Config path |
| `--offline` | false | Stub LLMs |
| `-v` (count) | 0 | Verbosity |

### `mags-codedev chat`

Interactive chat with the LLM about the codebase. Can read and write files.

```bash
mags-codedev chat
```

| Flag | Default | Description |
|---|---|---|
| `-c, --config` | auto | Config path |
| `--offline` | false | Stub LLMs |
| `-v` (count) | 0 | Verbosity |

### `mags-codedev tokens`

Display token usage and cost statistics across all models and runs.

```bash
mags-codedev tokens
```

Shows a table with: role, model, prompt tokens, completion tokens, total tokens, and cost.

### `mags-codedev list-models`

List available models from configured providers.

```bash
mags-codedev list-models
mags-codedev list-models -c /path/to/config.yaml
```

Probes each configured provider (OpenAI, Anthropic, Google, custom) and lists available models.

### `mags-codedev clean`

Remove all generated artifacts: `.mags-codedev/` directory, logs, worktrees, cache.

```bash
mags-codedev clean
mags-codedev clean --force    # skip confirmation
```

---

## Configuration

Config lives at `.mags-codedev/config.yaml` (default). Override with `-c`.

### Full Reference

```yaml
# API keys for LLM providers
api_keys:
  openai: "sk-..."
  anthropic: "sk-ant-..."
  gemini: "AIza..."

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
    # Multiple reviewers run concurrently
    reviewers:
      - provider: "openai"
        model: "gpt-4o"
      - provider: "google"
        model: "gemini-2.5-pro"
      - provider: "anthropic"
        model: "claude-3-5-sonnet-20240620"
      # Local model via OpenAI-compatible endpoint
      - provider: "custom_openai"
        model: "llama3:70b"
        base_url: "http://localhost:11434/v1"
        api_key: "ollama"

settings:
  language: "python"

  # Container runtime: auto | podman | docker | apptainer | singularity | local
  test_runner: "auto"

  # Container images (per runtime)
  docker_test_image: "mags-dev-env:latest"
  apptainer_test_image: "mags-dev-env.sif"

  # Python base image (optional override)
  python_base_image: "python:3.11-slim"

  # System packages to install in the container
  system_dependencies: ["build-essential", "pkg-config"]

  # Parallelism and budgets
  max_parallel_modules: 4
  max_test_fix_iterations: 5
  max_review_rounds: 3
  timeout_per_module_mins: 15

  # Artifact directory (auto-gitignored)
  base_dir: ".mags-codedev"

  # Logging: info | debug | trace
  log_level: "info"
```

### Custom OpenAI-Compatible Providers

For local models (Ollama, vLLM, LM Studio), use `provider: "custom_openai"`:

```yaml
reviewers:
  - provider: "custom_openai"
    model: "llama3:70b"
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"
```

---

## Architecture

### Build Graph Per Module

Each module has its own LangGraph with the following nodes:

```
coder_node → tester_node → log_checker_node → reviewer_node → router
     ^                                                    │
     └────────────────────────────────────────────────────┘
```

1. **Coder** — Generates or fixes module code. Receives: description, dependency source code, existing code, error summaries.
2. **Tester** — Generates or fixes tests. Runs tests in isolated container. Receives: module code, dependency code, existing tests, test results.
3. **LogChecker** — Analyzes test/lint output. Determines error location (`SOURCE_CODE` or `TEST_CODE`). Routes to the appropriate fixer.
4. **Reviewers** — Multiple LLM agents review code in parallel. Aggregate feedback is fed back to the Coder for revision.
5. **Router** — Decides the next step:
   - `coder_node` — code needs fixing
   - `tester_node` — tests need fixing
   - `reviewer_node` — code is correct, needs review
   - `__end__` — converged, all checks pass

### Parallelism

Modules with satisfied dependencies build **concurrently** in waves:

```
Wave 1:  [types.py] [config.py] [constants.py]     ← 3 parallel
Wave 2:  [validator.py] [utils.py]                  ← 2 parallel
Wave 3:  [engine.py]                                ← 1
Wave 4:  [pipeline.py] [cli.py]                     ← 2 parallel
```

Each module runs in its own:
- **Git worktree** (isolated from other modules)
- **LangGraph** (isolated state)
- **Container** (isolated test environment)

### Convergence Detection

A module is marked as converged when:
- Code hash matches previous iteration's code hash AND tests pass AND lint passes
- Test hash matches previous iteration's test hash (prevents test-only loops)
- Review rounds are exhausted or all reviewers approve

If either the code or tests change without progress, the module is marked as failed.

### Token Tracking

All LLM calls are tracked via `TokenCounter(BaseCallbackHandler)`. Usage is stored in the SQLite database and can be viewed with `mags-codedev tokens`.

---

## Test Environment

### Running Unit Tests

The package includes a pytest test suite:

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_coder_node.py -v

# Run with coverage
python3 -m pytest tests/ --cov=mags_codedev --cov-report=term-missing
```

### Container Test Environment

Module tests run in isolated containers. The framework:

1. Detects available container runtime (podman > docker > apptainer > singularity)
2. Builds a container image from the project's dependencies
3. Runs pytest/flake8/mypy inside the container
4. Mounts the worktree so the container sees the module code

To test locally without containers, set `test_runner: "local"` in config.

### Debugging Container Issues

```bash
# Check which runtime is detected
python3 -c "from mags_codedev.utils.docker_ops import _get_container_runtime; print(_get_container_runtime())"

# Check if the container image exists
podman images | grep mags-dev-env
docker images | grep mags-dev-env
```

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
- Cycles are detected and rejected before the build starts
- Modules with no dependencies build in the first wave
- Manifest description edits do **not** invalidate already-built modules (hash is based on `location` only)

---

## Troubleshooting

### "Module already built" but code is stale

The build system caches completed modules by their `location` hash. To force a rebuild:

```bash
mags-codedev build --force-fresh
```

### Container build fails

Check that your container runtime is installed and running:

```bash
podman info        # or: docker info
```

If the image fails to build, check `system_dependencies` in config.yaml.

### LLM API errors

Check `api_keys` in config.yaml. Verify the provider/model combination is valid:

```bash
mags-codedev list-models
```

### Module stuck in infinite loop

Check `max_test_fix_iterations` in config. If the code and tests are both changing but not converging, the module will be marked as failed after the iteration budget is exhausted. Use `mags-codedev debug` to investigate.

### Logs

Module logs are stored in `.mags-codedev/logs/`. Enable trace logging:

```bash
mags-codedev build -vv
```

---

## License

MIT
