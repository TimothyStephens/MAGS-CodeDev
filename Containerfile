# MAGS-CodeDev OMP Sandbox
#
# Build:  bash Containerfile_build
# Run:    omp-workspace

FROM node:22-bookworm-slim

ARG BUILD_VERSION="dev"
LABEL org.opencontainers.image.version="${BUILD_VERSION}"

# System deps
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
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://get.apptainer.com | sh -s -- -b /usr/local/bin

# Bun (required runtime for OMP)
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:${PATH}"

# OMP globally
RUN npm install -g @oh-my-pi/pi-coding-agent

# Python test/scientific packages
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip install pytest pytest-cov flake8 mypy bandit numpy pandas

# MAGS-CodeDev CLI (editable install for runtime edits)
COPY pyproject.toml /mags/pyproject.toml
COPY mags_codedev/ /mags/mags_codedev/
RUN pip install -e /mags

# OMP config dir (mounted at runtime as ~/.omp:/root/.omp)
RUN mkdir -p /root/.omp/agent

# Extension source
COPY mags-codedev-extension/ /ext/

# Link the extension into OMP (needs bun in PATH — set above)
RUN omp plugin link /ext

# Shell profile — sources .bashrc for login shells
RUN echo 'source /root/.bashrc 2>/dev/null' > /root/.profile

# .bashrc (aliases, prompt, PATH)
COPY .bashrc /root/.bashrc

# Git global config
RUN git config --global user.email "dev@mags-codedev.local" \
    && git config --global user.name "Dev"

# Podman rootless setup
RUN podman system reset --force 2>/dev/null || true

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
CMD ["-l"]
