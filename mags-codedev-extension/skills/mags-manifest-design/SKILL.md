---
name: mags-manifest-design
description: Use when creating or editing a MAGs-CodeDev manifest.json. Covers module decomposition, dependency modeling, and writing effective descriptions.
---

# MAGs-CodeDev Manifest Design

A manifest (`manifest.json`) is a JSON array of module specs. The build system computes a DAG from the `dependencies` field and builds modules in topological waves. Getting the manifest right is the single highest-leverage decision — the quality of the generated code depends on how well the modules are decomposed and described.

## Structure

```json
[
  {
    "location": "src/core/types.py",
    "description": "<what this module does, conventions, API>",
    "dependencies": []
  }
]
```

| Field | Required | Purpose |
|---|---|---|
| `location` | yes | Relative file path (e.g. `src/core/types.py`) |
| `description` | yes | Instructions given to the coder agent — be specific |
| `dependencies` | yes (can be `[]`) | Array of `location` values this module imports from |

## Decomposition principles

1. **One responsibility per module.** Each module should have a single, well-defined purpose. If a module does two unrelated things, split it.

2. **Bottom-up order.** Foundational modules (types, constants, config) have no dependencies. Domain logic depends on them. Entry points (CLI, pipeline) depend on everything below.

3. **Dependencies = imports.** A module's `dependencies` list every other manifest module it `import`s from. If module B uses a class defined in module A, B depends on A. The build system injects A's source code into B's prompt so the coder knows the interface.

4. **No cycles.** If A depends on B and B depends on A, the build will detect the cycle and abort. Refactor to break the cycle (extract shared code into a third module).

5. **Right granularity.** Too coarse (one module = whole subsystem) → the agent can't focus. Too fine (one module per function) → the agent can't see context. Aim for 50–300 lines per module.

## Writing effective descriptions

The `description` is the ONLY instruction the coder agent receives. It must be specific enough that a competent developer could write the module from it alone.

### Good description

```
"Input validation using pydantic v2 models. Define a PipelineConfig dataclass
with fields: name (str), version (str), timeout (int, default=30), retries
(int, default=3). Include a validate_config(config: dict) -> PipelineConfig
function that raises ValidationError on invalid input. Use frozen=True on the
dataclass for immutability. Follow PEP 8."
```

### Bad description

```
"Validation logic for the pipeline."
```

### What to include

- **What** the module does (its public API: classes, functions, signatures)
- **Key design decisions** (e.g., "use frozen=True", "raise ValidationError")
- **Naming conventions** (e.g., "prefix private helpers with _")
- **Edge cases to handle** (e.g., "return None for empty input")
- **Dependencies to use** (e.g., "use pydantic v2, not v1")

### What NOT to include

- Implementation details (the coder writes the code — you describe the interface)
- Boilerplate (the coder handles imports, `if __name__` blocks)
- Test expectations (the tester agent writes tests — describe the module's behavior)

## Example manifest

A pipeline project decomposed into 8 modules:

```json
[
  {
    "location": "src/core/constants.py",
    "description": "Project-wide constants: PIPELINE_VERSION='1.0', DEFAULT_TIMEOUT=30, MAX_RETRIES=3, EXIT_CODES dict mapping error types to int codes.",
    "dependencies": []
  },
  {
    "location": "src/core/types.py",
    "description": "Core data types using dataclasses: PipelineConfig (name, version, timeout, retries), StageResult (stage_name, status, duration_ms, output), PipelineResult (stages: list[StageResult], total_duration, success: bool). Use frozen=True on all dataclasses for immutability.",
    "dependencies": []
  },
  {
    "location": "src/core/config.py",
    "description": "Configuration loader. load_config(path: Path) -> PipelineConfig reads a YAML file and validates it. merge_defaults(config: dict) -> dict fills in missing keys with DEFAULT_TIMEOUT and MAX_RETRIES from constants. Raise FileNotFoundError if path doesn't exist.",
    "dependencies": ["src/core/types.py", "src/core/constants.py"]
  },
  {
    "location": "src/validation/validator.py",
    "description": "Pipeline input validator using pydantic v2. ValidateConfig model with the same fields as PipelineConfig. validate_input(data: dict) -> ValidateConfig raises ValidationError with field-specific messages. validate_config_object(config: PipelineConfig) -> bool checks runtime constraints (timeout > 0, retries <= 10).",
    "dependencies": ["src/core/types.py"]
  },
  {
    "location": "src/logging/pipeline_logger.py",
    "description": "Structured logger for pipeline execution. get_logger(name: str, level: str='INFO') -> logging.Logger configures a handler that writes JSON lines to stderr. log_stage_result(result: StageResult) logs stage name, status, and duration. Uses constants from src/core/constants.py for log level defaults.",
    "dependencies": ["src/core/constants.py"]
  },
  {
    "location": "src/transformers/base.py",
    "description": "Abstract base class for pipeline stage transformers. TransformerBase with abstract transform(data: dict) -> dict and validate(data: dict) -> bool methods. TransformerRegistry singleton that registers and looks up transformers by name. Raise KeyError for unknown transformer names.",
    "dependencies": ["src/core/types.py", "src/validation/validator.py"]
  },
  {
    "location": "src/engine/pipeline.py",
    "description": "Pipeline execution engine. Pipeline class with run(config: PipelineConfig) -> PipelineResult method that loads transformers from the registry, executes each stage, catches exceptions, and aggregates results. track_stage decorator wraps transform calls with timing and logging. Retry failed stages up to config.retries times.",
    "dependencies": ["src/core/types.py", "src/core/config.py", "src/transformers/base.py", "src/logging/pipeline_logger.py"]
  },
  {
    "location": "src/engine/scheduler.py",
    "description": "DAG-aware scheduler for parallel pipeline stages. Scheduler class with add_stage(name, transformer, deps: list[str]) and execute() -> PipelineResult methods. Uses asyncio.gather for stages with satisfied dependencies. Raises RuntimeError if a cycle is detected.",
    "dependencies": ["src/engine/pipeline.py", "src/core/types.py", "src/logging/pipeline_logger.py"]
  },
  {
    "location": "src/cli/main.py",
    "description": "CLI entry point using typer. Commands: run (load config, execute pipeline), validate (check config file), list-stages (show registered transformers). Uses src/core/config.py for config loading and src/engine/scheduler.py for execution. Print results as a rich table.",
    "dependencies": ["src/engine/pipeline.py", "src/engine/scheduler.py", "src/core/config.py", "src/logging/pipeline_logger.py", "src/transformers/base.py"]
  }
]
```

## Tips

- Order doesn't matter — the build system computes topological order.
- Editing a description auto-invalidates that module on the next build (spec-aware rebuild).
- Use `mags_build --module <location>` to rerun a single module after editing its spec.
- Modules with no dependencies build in the first wave (in parallel).
- The tester agent sees the module code + dependency source — it can test cross-module integration.
- Keep descriptions in plain English (no JSON, no code blocks) — the coder agent writes the actual code.
