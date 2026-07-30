# MAGs-CodeDev OMP Extension

Custom tools for [Oh My Pi](https://omp.sh) that expose the [MAGs-CodeDev](https://github.com/j0k3r/MAGs-CodeDev) multi-agent CLI as structured OMP tools with streaming JSONL progress.

## What it does

Wraps the `mags-codedev` CLI in OMP's `registerTool` API so the agent can call each command with structured parameters, receive typed output, and see live build progress in the TUI instead of raw `bash` invocations.

Key improvements over the old extension:

- **Streaming JSONL**: `mags_build` defaults to `json=true`, which passes `--json` to the CLI and parses JSONL status events. Progress is streamed via `onUpdate` so the OMP TUI shows live module status.
- **Per-task rerun**: `mags_build` accepts `module` to rerun a single task after a spec edit or manual fix.
- **Updated signatures**: matches the new CLI (no `--parallelism`/`--max-iterations`; uses config-driven `max_parallel_modules`/`max_test_fix_iterations`).
- **Session auto-detect**: `session_start` hook detects a MAGs-CodeDev workspace and notifies the user.

## Prerequisites

- `mags-codedev` installed and on `PATH`
- OMP installed (`bun install -g @oh-my-pi/pi-coding-agent`)

## Installation

```bash
# Project scope (recommended — lives in your repo)
omp install ./mags-codedev-extension

# Or global scope
omp install -g ./mags-codedev-extension
```

Verify:

```bash
omp -p '/extensions'
```

Should show `mags-codedev-extension` loaded with 7 tools.

## Tools

| Tool | CLI equivalent | Key params | Description |
|---|---|---|---|
| `mags_init` | `mags-codedev init` | `interactive` | Initialize workspace |
| `mags_build` | `mags-codedev build` | `module`, `force_fresh`, `json`, `verbose` | Run build loop (streaming JSONL by default) |
| `mags_test` | `mags-codedev test` | `verbose` | Project-wide test run |
| `mags_debug` | `mags-codedev debug` | `error_msg`, `module_location` | Auto-fix from error trace |
| `mags_tokens` | `mags-codedev tokens` | — | Token usage stats |
| `mags_list_models` | `mags-codedev list-models` | `config_path` | Available models |
| `mags_clean` | `mags-codedev clean` | `force` | Remove artifacts |

## JSONL event schema

When `mags_build` is called with `json=true` (default), the CLI emits JSONL to stdout. Each line is a JSON object with a `ts` (ISO-8601 UTC) and `event` field:

| Event | Fields | When |
|---|---|---|
| `build_start` | `manifest`, `total_modules`, `already_built` | Before the first wave |
| `module_start` | `location`, `hash`, `session` | A module begins |
| `module_step` | `location`, `step`, `iteration` | A graph node executes |
| `module_tokens` | `location`, `tokens_in`, `tokens_out` | Token usage recorded |
| `module_end` | `location`, `status`, `iterations`, `log_file`, `tokens_in`, `tokens_out` | Module finished |
| `wave_end` | `succeeded`, `failed` | DAG wave completed |
| `build_end` | `total_modules`, `succeeded`, `failed`, `blocked`, `tokens_in`, `tokens_out` | Final summary |

The extension parses these and streams `module_step` events to the OMP TUI via `onUpdate`. The final `build_end` event is returned as structured `details` so the agent can act on failures programmatically.

## Container usage

Build a container with both OMP and MAGs-CodeDev:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://bun.sh/install | bash
ENV PATH=$PATH:/root/.bun/bin
RUN bun install -g @oh-my-pi/pi-coding-agent

WORKDIR /mags
COPY pyproject.toml .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir .

WORKDIR /ext
COPY mags-codedev-extension/ .
RUN omp install .

WORKDIR /workspace
ENTRYPOINT ["omp"]
```
