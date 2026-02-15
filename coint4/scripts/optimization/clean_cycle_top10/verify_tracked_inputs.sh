#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage:
  verify_tracked_inputs.sh [path/to/tracked_inputs.txt]

Checks that each listed path is tracked by git (required for SYNC_UP=1).
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"
cd "$repo_root"

list_file="${1:-coint4/configs/clean_cycle_top10/tracked_inputs_20260215_clean_top10.txt}"
if [[ ! -f "$list_file" ]]; then
  echo "ERROR: list file not found: $list_file" >&2
  exit 2
fi

missing=0
while IFS= read -r p; do
  # Skip comments/blank lines
  [[ -z "$p" ]] && continue
  [[ "$p" == \#* ]] && continue

  if ! git ls-files --error-unmatch -- "$p" >/dev/null 2>&1; then
    echo "MISSING_TRACKED: $p" >&2
    missing=1
  fi
done < "$list_file"

exit "$missing"
