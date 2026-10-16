#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FDS_BIN="${FDS_BIN:-fds}"
OUTPUT_ROOT="${1:-$REPO_ROOT/artifacts/p8_v2_fds_raw}"

command -v "$FDS_BIN" >/dev/null 2>&1 || {
  echo "FDS executable not found. Install the pinned official release or set FDS_BIN." >&2
  exit 2
}

mkdir -p "$OUTPUT_ROOT"
"$FDS_BIN" -v | tee "$OUTPUT_ROOT/fds_version.txt"
for input in "$REPO_ROOT"/fds/p8_v2/*.fds; do
  case_name="$(basename "$input" .fds)"
  case_dir="$OUTPUT_ROOT/$case_name"
  mkdir -p "$case_dir"
  cp "$input" "$case_dir/"
  (cd "$case_dir" && "$FDS_BIN" "$(basename "$input")")
done

echo "Raw synthetic FDS outputs written to $OUTPUT_ROOT"
echo "Convert them to the documented NPZ/manifest contract before P8 ingestion."
