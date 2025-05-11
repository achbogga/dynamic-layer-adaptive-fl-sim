# dynamic-layer-adaptive-fl-sim

A simulation framework for Dynamic Layer‑Adaptive Federated Learning (DLA‑AI).

## Features
- Top-down cloud–fog–edge FL simulation
- Dynamic task scheduling and SLA-driven fault tolerance
- Layer-wise model partitioning and communication-efficient updates
- Containerized setup with UV, Docker, and VS Code Dev Container

## Recommended Setup (VS Code Dev Container)

**The recommended way to develop and run this project is using [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers).**

1. Install [Docker](https://www.docker.com/get-started).
2. Install [Visual Studio Code](https://code.visualstudio.com/) and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
3. Open this repository in VS Code and select **"Reopen in Container"** when prompted.

This will automatically build the development environment with all dependencies.

## Alternative: Local Installation

### Prerequisites
- [UV](https://docs.astral.sh/uv): Python environment manager
- Docker (optional, for containerized runs)
- VS Code with Remote - Containers extension (optional)

### Using UV
```bash
pip install uv
uv sync --locked
```

## Run the Simulation with all the experiments from the configs as described in the paper
```bash
bash run_experiments.sh
```
