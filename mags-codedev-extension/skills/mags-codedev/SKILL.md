# MAGs-CodeDev

Multi-agent LLM code generation workflow. Initializes a workspace, generates code and tests in isolated containers, runs lint/type-check/security scans, and iterates until all modules pass.

## Workflow

1. **`mags_init`** — Create AGENT.md, manifest.json, .gitignore, and SQLite database. Use `interactive=false` for non-interactive mode.

2. **`mags_build`** — Run the full build loop:
   - Coder agent generates code from specs
   - Tester agent writes pytest unit tests
   - Runs pytest + coverage, flake8, mypy, bandit in isolated environment
   - Log checker analyzes failures, routes fix to coder or tester
   - Multi-LLM review before marking module complete
   - Repeats until all modules pass or `max_iterations` reached

3. **`mags_test`** — Run pytest for all modules in the isolated environment.

4. **`mags_debug`** — Pass an error trace to the LLM for automatic fixing. Accepts a log file path for auto-detection.

5. **`mags_tokens`** — Show token usage by role and model.

6. **`mags_list_models`** — List available models from configured providers.

7. **`mags_clean`** — Remove cache, logs, and worktrees.

## Key Files

- `manifest.json` — Module definitions (location, description, dependencies). Created by `init` or manually.
- `config.yaml` — LLM provider config, API keys, Docker/Apptainer runner settings.
- `AGENT.md` — Instructions for the coding agents (language, coding standards).
- `requirements.txt` — Project runtime dependencies (NOT dev/test tools — those are injected at build time).

## Tips

- Use `mags_build` for the full pipeline; `mags_test` for a quick project-wide test run outside the loop.
- The `debug` tool auto-detects the module from log file hash — you only need `module_location` when passing raw error text.
- Coverage reports (`pytest-cov`) show which lines weren't exercised — use this in `mags_debug` to improve test quality.
- Bandit skips B101 (assert_used) and B104 (hardcoded_bind_all_interfaces) to reduce false positives.
