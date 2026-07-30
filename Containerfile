# MAGS-CodeDev OMP Sandbox
#
# Build:  bash Containerfile_build
# Run:    omp-workspace

FROM node:22-bookworm-slim

ARG BUILD_VERSION="dev"
LABEL org.opencontainers.image.version="${BUILD_VERSION}"

# ── System deps ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git \
    curl ca-certificates unzip \
    gcc g++ make cmake \
    rustc cargo \
    golang-go \
    docker.io \
    podman \
    sudo \
    nano vim \
    fd-find \
    ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://get.apptainer.com | sh -s -- -b /usr/local/bin

# fd is installed as fdfind on Debian — symlink to fd for pi-nvim-bridge
RUN ln -sf /usr/bin/fdfind /usr/bin/fd

# ── Neovim 0.11+ (for pi-nvim-bridge) ────────────────────────────
RUN curl --http1.1 --retry 3 --retry-delay 2 -fsSL \
        -o /tmp/nvim.tar.gz \
        https://github.com/neovim/neovim/releases/download/stable/nvim-linux-x86_64.tar.gz \
    && tar xzf /tmp/nvim.tar.gz -C /opt \
    && rm -f /tmp/nvim.tar.gz \
    && ln -sf /opt/nvim-linux-x86_64/bin/nvim /usr/bin/nvim

# ── Bun (required runtime for OMP) ───────────────────────────────
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:${PATH}"

# ── OMP globally ─────────────────────────────────────────────────
RUN npm install -g @oh-my-pi/pi-coding-agent

# ── Python packages — split for Docker layer caching + retry ─────
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_DEFAULT_TIMEOUT=120
RUN pip install --retries 3 pytest pytest-cov flake8 mypy bandit
RUN pip install --retries 3 pyright
RUN pip install --retries 3 numpy pandas
RUN pip install --retries 3 hound-mcp[all]
RUN npm install -g playwright && playwright install chromium

# ── pi-nvim-bridge — OMP autocomplete in Neovim ──────────────────
RUN git clone --depth 1 https://github.com/dabstractor/pi-nvim-bridge.git /opt/pi-nvim-bridge \
    && cd /opt/pi-nvim-bridge \
    && omp plugin link . \
    && mkdir -p /root/.config/pi-bridge \
    && echo 'require("pi-bridge").setup({})' > /root/.config/pi-bridge/init.lua

# ── pi-codegraph — structural code analysis ──────────────────────
RUN omp install @isac322/pi-codegraph || true

# ── Context7 MCP — up-to-date library documentation ──────────────
RUN mkdir -p /root/.omp/agent
RUN echo '{"mcpServers":{"context7":{"command":"npx","args":["-y","@upstash/context7-mcp"]}}}' \
    > /root/.omp/agent/.mcp.json

# ── Code quality & workflow extensions ────────────────────────────
RUN omp install pi-lens || true
RUN omp install pi-context-prune || true
RUN omp install @code-yeongyu/pi-rules || true
RUN omp install @narumitw/pi-statusline || true

# ── Security & permission extensions ─────────────────────────────
RUN omp install @aliou/pi-guardrails || true
RUN omp install @gotgenes/pi-permission-system || true

# ── MAGS-CodeDev CLI ─────────────────────────────────────────────
# Includes the new modular structure: cmds/, agents/, backends/, utils/
COPY pyproject.toml /mags/pyproject.toml
COPY requirements.txt /mags/requirements.txt
COPY mags_codedev/ /mags/mags_codedev/
COPY config.template.yaml /mags/config.template.yaml
RUN pip install -e /mags

# ── MAGs-CodeDev OMP extension ───────────────────────────────────
COPY mags-codedev-extension/ /ext/
RUN omp plugin link /ext

# ── Shell profile ────────────────────────────────────────────────
RUN echo 'source /root/.bashrc 2>/dev/null' > /root/.profile

# ── .bashrc ──────────────────────────────────────────────────────
COPY .bashrc /root/.bashrc

# ── Git global config ────────────────────────────────────────────
RUN git config --global user.email "dev@mags-codedev.local" \
    && git config --global user.name "Dev"

# ── Podman rootless setup ────────────────────────────────────────
RUN podman system reset --force 2>/dev/null || true

# ── Environment ──────────────────────────────────────────────────
ENV EDITOR=nvim
ENV VISUAL=nvim
ENV PI_NVIM_APPNAME=pi-bridge

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
CMD ["-l"]
