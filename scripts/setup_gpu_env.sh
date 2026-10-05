#!/usr/bin/env bash
# One-time: install a prebuilt CUDA torch wheel next to the Nix python (see flake.nix `gpu` shell).
#   nix develop .#gpu --command bash scripts/setup_gpu_env.sh [cu130|cu128]
set -euo pipefail
CUDA="${1:-cu130}"
TARGET="$PWD/.venv-gpu/lib/python3.14/site-packages"
mkdir -p "$TARGET"
uv pip install --python "$(which python3)" --target "$TARGET" \
  --index-url "https://download.pytorch.org/whl/$CUDA" torch
