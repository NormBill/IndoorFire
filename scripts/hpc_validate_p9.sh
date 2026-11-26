#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/run_p9_postarrival_intervention.py

