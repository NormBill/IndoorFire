#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/run_p8_registered_closed_loop.py \
  --config configs/p8_registered_closed_loop.yaml \
  --output artifacts/p8_registered_closed_loop
python scripts/run_p8_v2_preentry_ingress.py \
  --config configs/p8_v2_preentry_ingress.yaml \
  --output artifacts/p8_v2_preentry_ingress
