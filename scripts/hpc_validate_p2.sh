#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Run inside the cluster's project environment after synchronizing the repo.
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/run_p2_fire_simulation.py
