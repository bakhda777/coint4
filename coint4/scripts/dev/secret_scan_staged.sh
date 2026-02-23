#!/usr/bin/env bash
set -euo pipefail

# Simple staged diff guardrail:
# - blocks committing forbidden paths (secrets/artifacts/logs)
# - blocks obvious credential patterns in staged diffs
#
# Intentionally does NOT print matching diff lines.

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

fail=0

forbidden_path_re='^(\.secrets/|coint4/artifacts/|coint4/outputs/|outputs/|\.ralph-tui/iterations/).*$'
forbidden_ext_re='(\.pid|\.log)$'

mapfile -t staged_paths < <(git diff --cached --name-only)

for p in "${staged_paths[@]}"; do
  # Allow-list small, intentionally-tracked WFA aggregate metadata under coint4/artifacts.
  # This keeps heavy run outputs (wfa/runs) blocked, while letting us commit queues/search spaces/rollups.
  if [[ "$p" =~ ^coint4/artifacts/ ]]; then
    if [[ "$p" =~ ^coint4/artifacts/wfa/aggregate/[^/]+/run_queue\.csv$ ]]; then
      continue
    fi
    if [[ "$p" =~ ^coint4/artifacts/wfa/aggregate/[^/]+/search_space\.(csv|md)$ ]]; then
      continue
    fi
    if [[ "$p" =~ ^coint4/artifacts/wfa/aggregate/rollup/(README\.md|run_index\.(csv|json|md))$ ]]; then
      continue
    fi
  fi

  if [[ "$p" =~ $forbidden_path_re ]]; then
    echo "[secret-scan] ERROR: forbidden staged path: $p" >&2
    fail=1
  fi
  if [[ "$p" =~ $forbidden_ext_re ]]; then
    echo "[secret-scan] ERROR: forbidden staged file extension: $p" >&2
    fail=1
  fi
done

# Content checks (do not print matching lines)
content_patterns=(
  'X-API-KEY'
  'SERVSPACE_API_KEY='
  'BYBIT_API_KEY='
  'BYBIT_API_SECRET='
  '[0-9a-fA-F]{64}'
)

for p in "${staged_paths[@]}"; do
  diff_text="$(git diff --cached -U0 -- "$p" || true)"
  for re in "${content_patterns[@]}"; do
    if printf '%s' "$diff_text" | grep -Eq "$re"; then
      echo "[secret-scan] ERROR: sensitive pattern detected in staged diff for: $p" >&2
      echo "[secret-scan]        pattern: $re" >&2
      fail=1
    fi
  done
done

if [[ "$fail" -ne 0 ]]; then
  echo "[secret-scan] Aborting commit." >&2
  exit 1
fi

echo "[secret-scan] OK"
