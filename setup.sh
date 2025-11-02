#!/usr/bin/env bash
set -euox pipefail

apt update && apt install -y libnuma-dev tmux docker.io

grep -qxF 'export HF_HOME=/workspace/hf' ~/.bashrc || echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc
mkdir -p /workspace/hf

curl -LsSf https://astral.sh/uv/install.sh | sh

export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"

command -v uv

uv venv -p 3.12 --seed --clear

uv run --no-sync bash examples/env/setup-pip-deps.sh
uv pip install "mcp[cli]" reasoning-gym anthropic math_verify fastmcp
uv pip uninstall pynvml