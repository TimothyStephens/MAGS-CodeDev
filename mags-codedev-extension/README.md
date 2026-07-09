# MAGs-CodeDev OMP Extension

Custom tools for [Oh My Pi](https://omp.sh) that expose the [MAGs-CodeDev](https://github.com/j0k3r/MAGs-CodeDev) multi-agent CLI as structured OMP tools.

## What it does

Wraps the `mags-codedev` CLI in OMP's `registerTool` API so the agent can call each command with structured parameters, receive typed output, and surface tool cards in the TUI instead of raw `bash` invocations.

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

| Tool | CLI equivalent | Description |
| --- | --- | --- |
| `mags_init` | `mags-codedev init` | Initialize workspace |
| `mags_build` | `mags-codedev build` | Run full build loop |
| `mags_test` | `mags-codedev test` | Project-wide test run |
| `mags_debug` | `mags-codedev debug` | Auto-fix from error trace |
| `mags_tokens` | `mags-codedev tokens` | Token usage stats |
| `mags_list_models` | `mags-codedev list-models` | Available models |
| `mags_clean` | `mags-codedev clean` | Remove artifacts |

## Container usage

Build a container with both OMP and MAGs-CodeDev:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# OMP via Bun
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH=$PATH:/root/.bun/bin
RUN bun install -g @oh-my-pi/pi-coding-agent

# MAGs-CodeDev
WORKDIR /mags
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir .

# Extension
WORKDIR /ext
COPY mags-codedev-extension/ .
RUN omp install .

WORKDIR /workspace
ENTRYPOINT ["omp"]
```

Run:

```bash
docker run -it --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $PWD:/workspace \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    mag-omp
```
