#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/validate_gate1_gate2.py
