FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Git and Docker CLI (required for branching and DooD)
RUN apt-get update && \
    apt-get install -y git docker.io && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies first to leverage Docker layer caching
COPY pyproject.toml .
RUN pip install --no-cache-dir build && \
    pip install --no-cache-dir .

# Copy the rest of the application
COPY mags_codedev/ mags_codedev/
COPY README.md .

# The entrypoint allows the user to run the container as if it were the CLI tool
ENTRYPOINT ["mags-codedev"]

# Default command if none is provided
CMD ["--help"]