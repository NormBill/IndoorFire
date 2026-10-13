#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Run inside the cluster's project environment after synchronizing the repo.
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/run_p2_fire_simulation.py
python scripts/run_p3_assimilation_demo.py
python scripts/run_p4_hazard_demo.py
python scripts/run_p5_weight_sensitivity.py
python scripts/run_p5_planning_demo.py
