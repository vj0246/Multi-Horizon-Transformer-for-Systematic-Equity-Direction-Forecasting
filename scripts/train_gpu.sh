#!/usr/bin/env bash
# Train on the GPU inside WSL2. Usage (from Windows):
#   wsl -d Ubuntu bash "/mnt/c/.../scripts/train_gpu.sh" [module]
# Defaults to the cross-sectional track.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/wsl_gpu_env.sh"
cd "$HERE/.."
python -m "${1:-Source.Backtest.run_cross_section}"
