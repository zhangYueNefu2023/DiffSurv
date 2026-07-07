#!/usr/bin/env bash
set -euo pipefail

python scripts/preprocess_nested.py \
  --raw-data-dir raw_data \
  --output-dir processed_data \
  --num-folds 5 \
  --variance-ratio 0.80

python scripts/train.py --config configs/default.yaml

python scripts/evaluate.py --config configs/default.yaml

python scripts/evaluate_missingness.py \
  --config configs/default.yaml \
  --rates 0.0 0.1 0.2 0.3 0.5 0.7 0.8 0.9
