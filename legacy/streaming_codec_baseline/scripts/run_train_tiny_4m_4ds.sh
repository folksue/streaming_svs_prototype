#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python train.py \
  --config configs/base/runtime_tiny.yaml \
  --data-config configs/data/mix_4ds_template.yaml \
  --model-config configs/model/tiny_4m.yaml
