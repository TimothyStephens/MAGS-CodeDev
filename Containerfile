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
    curl ca-certificates unzip gnupg \
    gcc g++ make cmake \
    rustc cargo \
    golang-go \
    docker.io \
    podman \
    sudo \
    nano vim \
    ripgrep \
    # Playwright Chromium dependencies
    libnspr4 libnss3 \
    libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0 \
    libdbus-1-3 libx11-6 libx11-xcb1 libxcomposite1 libxcursor1 \
    libxdamage1 libxext6 libxfixes3 libxrandr2 libxrender1 \
    libxss1 libxtst6 libgbm1 libxcb1 libxkbcommon0 \
    libasound2 libpulse0 libcups2 \
    libpango-1.0-0 libcairo2 libdrm2 libwayland-client0 \
    libxshmfence1 fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# fd is installed as fdfind on Debian — symlink to fd for pi-nvim-bridge
RUN ln -sf /usr/bin/fdfind /usr/bin/fd

# ── Neovim 0.11+ (for pi-nvim-bridge) ────────────────────────────
# Use apt for reliability over GitHub releases download
RUN curl -fsSL https://github.com/neovim/neovim/releases/download/v0.10.4/nvim-linux-x86_64.tar.gz \
        -o /tmp/nvim.tar.gz 2>/dev/null \
    && tar xzf /tmp/nvim.tar.gz -C /opt \
    && rm -f /tmp/nvim.tar.gz \
    && ln -sf /opt/nvim-linux-x86_64/bin/nvim /usr/bin/nvim \
    || { \
        apt-get update && apt-get install -y --no-install-recommends neovim \
        && rm -rf /var/lib/apt/lists/*; \
    }

# ── Bun (required runtime for OMP) ───────────────────────────────
RUN for i in 1 2 3; do curl -fsSL https://bun.sh/install | bash && break || sleep 15; done
ENV PATH="/root/.bun/bin:${PATH}"

# ── OMP globally ─────────────────────────────────────────────────
RUN npm install -g --retry 5 --retries 10 @oh-my-pi/pi-coding-agent

# ── Python packages ──────────────────────────────────────────────
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10
RUN pip install --retries 10 --timeout 120 \
        pytest pytest-cov flake8 mypy bandit pyright numpy pandas playwright
RUN pip install --retries 10 --timeout 120 hound-mcp[all]
# Persist Playwright browsers to volume-mounted /config/ (not ephemeral /root/)
ENV PLAYWRIGHT_BROWSERS_PATH=/config/.cache/ms-playwright
RUN mkdir -p "$PLAYWRIGHT_BROWSERS_PATH" \
    && (command -v playwright &>/dev/null && echo "playwright already installed" || npm install -g --force playwright) && \
    playwright install chromium

# ── pi-nvim-bridge — OMP autocomplete in Neovim ──────────────────
RUN for i in 1 2 3 4 5; do git clone --depth 1 https://github.com/dabstractor/pi-nvim-bridge.git /opt/pi-nvim-bridge && break || sleep 10; done \
    && cd /opt/pi-nvim-bridge \
    && omp plugin link . \
    && mkdir -p /root/.config/pi-bridge \
    && echo 'require("pi-bridge").setup({})' > /root/.config/pi-bridge/init.lua

# ── pi-codegraph — structural code analysis ──────────────────────
RUN omp install @isac322/pi-codegraph || true

# ── Code quality & workflow extensions ────────────────────────────
RUN omp install pi-lens || true
RUN omp install pi-context-prune || true
RUN omp install pi-rules || true
RUN omp install @narumitw/pi-statusline || true

# ── Security & permission extensions ─────────────────────────────
RUN cd /root/.omp/plugins && npm install --retry 5 --retries 10 @aliou/pi-guardrails@latest
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

# ── Relink plugins (may get unlinked by subsequent npm installs) ──
RUN cd /opt/pi-nvim-bridge && omp plugin link . \
    && omp plugin link /ext

# ── Context7 MCP — up-to-date library documentation ──────────────
# Written last to avoid being overwritten by OMP plugin installs
RUN mkdir -p /root/.omp/agent \
    && echo '{"mcpServers":{"context7":{"command":"npx","args":["-y","@upstash/context7-mcp"]}}}' \
        > /root/.omp/agent/.mcp.json

# ── Shell profile ────────────────────────────────────────────────
RUN echo 'source /root/.bashrc 2>/dev/null' > /root/.profile

# ── .bashrc ──────────────────────────────────────────────────────
COPY .bashrc /root/.bashrc

# ── Git global config ────────────────────────────────────────────
RUN git config --global user.email "dev@mags-codedev.local" \
    && git config --global user.name "Dev"

# ── Podman rootless setup ────────────────────────────────────────
# ── Environment ──────────────────────────────────────────────────
ENV EDITOR=nvim
ENV VISUAL=nvim
ENV PI_NVIM_APPNAME=pi-bridge

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
CMD ["-l"]
