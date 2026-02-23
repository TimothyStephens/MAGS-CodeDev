# MAGs-CodeDev

**M**ulti-**A**gent **G**raph **S**ystem for **Code** **Dev**elopment.

MAGs-CodeDev is an autonomous, multi-agent AI software engineer built for the command line. It takes a functional manifest, spins up parallel LLM agents via LangGraph, generates code, writes edge-case unit tests, executes them in isolated Docker containers, self-heals based on logs, and merges the completed code via Git worktrees.

## Features
- 🚀 **Parallel Execution**: Builds multiple functions concurrently using isolated Git worktrees.
- 🐳 **Dockerized Testing**: Runs generated unit tests in ephemeral, isolated containers.
- 🧠 **Multi-Agent Consensus**: Utilizes different LLMs (OpenAI, Anthropic, Gemini) for coding, testing, log-checking, and final peer review.
- 📊 **Token Tracking**: Logs and displays rich terminal tables tracking token usage and costs.

## Prerequisites
- Python 3.10+
- Git
- Docker Desktop / Docker Engine running

## Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/MAGs-CodeDev.git](https://github.com/yourusername/MAGs-CodeDev.git)
cd MAGs-CodeDev

# Install the package globally or in a virtual environment
pip install -e .