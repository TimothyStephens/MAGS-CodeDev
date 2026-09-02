---
name: mags-codedev
description: Use when working with a MAGs-CodeDev workspace — building modules, debugging failures, checking token usage, listing models, and cleaning artifacts.
---

# MAGs-CodeDev

Multi-agent LLM code generation workflow using LangGraph. Initializes a workspace, generates code and tests in isolated containers, runs lint/type-check/security scans, iterates with self-healing, and merges via git worktrees.

## Workflow

1. **`mags_init`** — Create AGENT.md, manifest.json, .gitignore, SQLite database, and config. Pass `interactive=false` for non-interactive mode (skips the AI Architect session).

2. **`mags_build`** — Run the full build loop:
   - Modules are built in DAG dependency order (topological waves).
   - Each module runs in its own git worktree + container.
   - Coder → Tester → run tests → LogChecker (diagnose) → Reviewers (multi-LLM consensus).
   - If a module fails, independent modules in later waves still run.
   - Use `--module <location>` to rerun a single task after editing its spec or fixing a failure.
   - Use `json=true` (default) for streaming JSONL progress in the OMP TUI.
   - Use `force_fresh=true` to rebuild everything from scratch.

3. **`mags_test`** — Run pytest for all modules in the container environment.

4. **`mags_debug`** — Pass an error trace or bug description to the LLM for automatic fixing. Accepts a log file path (auto-detects the module from its hash); when a module is resolved the full fix loop runs automatically, otherwise only the trace is analyzed — then rerun via `mags_build --module <location>`.

5. **`mags_tokens`** — Show token usage by role and model.

6. **`mags_list_models`** — List available models from configured providers.

7. **`mags_clean`** — Remove cache, logs, and worktrees.

## Key files

- `manifest.json` — Module definitions (location, description, dependencies). Created by `init` or the AI Architect.
- `config.yaml` — LLM provider config, API keys, container runtime settings, budgets.
- `AGENT.md` — Instructions for the coding agents (language, coding standards).
- `requirements.txt` — Project runtime dependencies (test tools injected at build time).

## Spec-aware rebuilds

Editing a module's `description` or `dependencies` in the manifest automatically invalidates that module on the next `mags_build` — you don't need `force_fresh`. The location-based hash is preserved for log/worktree continuity.

## Failure inspection

Failed-task logs (`.mags-codedev/logs/<hash>.log`) contain the full LLM conversation (prompts + responses) with file contents elided at INFO level. Use `verbose=1` (debug) to see full file contents in the log. Use `mags_debug` with the log file path to auto-detect the module and attempt a fix.

## LLM reliability

- Hard LLM failures (auth, quota) fail the task with a real error — no silent stub fallbacks (the log checker falls back to keyword analysis with a logged warning if its LLM call fails).
- Reviewer failures are neutral (skip = not a vote). Approval requires a strict majority of all configured reviewers.
- Transient errors (rate limits, 503s) are retried with exponential backoff.

## Tips

- Use `mags_build --module <loc>` to rerun one task after a manual fix or spec edit.
- Use `mags_build force_fresh=true` only when you want to nuke everything.
- The `mags_debug` tool auto-detects the module from log file hash — you only need `module_location` when passing raw error text.
