#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_train_100m.sh [CONFIG_PATH]
#
# Example:
#   bash scripts/run_train_100m.sh config.yaml

CONFIG_PATH="${1:-config.yaml}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

echo "[1/2] Preprocess caches with config: ${CONFIG_PATH}"
python3 preprocess_encodec.py --config "${CONFIG_PATH}" --split both

echo "[2/2] Start training with config: ${CONFIG_PATH}"
python3 train.py --config "${CONFIG_PATH}"

