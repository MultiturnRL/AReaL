set -eoux pipefail

echo "export HF_HOME=/workspace/hf" >> ~/.bashrc

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv -p 3.12 --seed --clear
uv run --no-sync bash examples/env/setup-pip-deps.sh
uv pip install "mcp[cli]" reasoning-gym anthropic math_verify fastmcp
