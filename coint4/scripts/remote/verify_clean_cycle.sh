#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: verify_clean_cycle.sh [--sync-up]

Checks:
  - No tracked files under known-heavy/generated paths (e.g. wfa/runs, outputs).
  - No staged files under known-heavy/generated paths.
  - No staged "large" files (default threshold: 10MB) to avoid accidental heavy commits.

With --sync-up:
  - Fails if there are untracked files under paths that must be synced to VPS via SYNC_UP=1
    (because SYNC_UP=1 uses git ls-files and will NOT sync untracked files).

EOF
}

SYNC_UP_MODE=0
if [[ $# -gt 1 ]]; then
  usage
  exit 2
fi
if [[ $# -eq 1 ]]; then
  if [[ "$1" == "--sync-up" ]]; then
    SYNC_UP_MODE=1
  else
    usage
    exit 2
  fi
fi

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${ROOT}" || ! -d "${ROOT}/.git" ]]; then
  echo "ERROR: must be run inside a git repo." >&2
  exit 1
fi
cd "${ROOT}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

echo "[check] repo_root=${ROOT}"

# 1) Tracked heavy/generated paths must be empty.
TRACKED_FORBIDDEN=(
  "coint4/artifacts/wfa/runs"
  "coint4/artifacts/wfa/traces"
  "coint4/artifacts/baseline_traces"
  "coint4/outputs"
  "outputs"
  ".ralph-tui/iterations"
)
for p in "${TRACKED_FORBIDDEN[@]}"; do
  if git ls-files "${p}" --error-unmatch >/dev/null 2>&1; then
    fail "tracked forbidden path detected: ${p}"
  fi
  if [[ -n "$(git ls-files "${p}/**" 2>/dev/null || true)" ]]; then
    fail "tracked forbidden files detected under: ${p}/"
  fi
done
echo "[ok] no tracked forbidden paths"

# 2) Staged heavy/generated paths must be empty.
STAGED="$(git diff --cached --name-only)"
if [[ -n "${STAGED}" ]]; then
  while IFS= read -r f; do
    case "${f}" in
      coint4/artifacts/wfa/runs/*|coint4/artifacts/wfa/runs/**) fail "staged forbidden: ${f}" ;;
      coint4/artifacts/wfa/traces/*|coint4/artifacts/wfa/traces/**) fail "staged forbidden: ${f}" ;;
      coint4/artifacts/baseline_traces/*|coint4/artifacts/baseline_traces/**) fail "staged forbidden: ${f}" ;;
      coint4/outputs/*|coint4/outputs/**) fail "staged forbidden: ${f}" ;;
      outputs/*|outputs/**) fail "staged forbidden: ${f}" ;;
      .ralph-tui/iterations/*|.ralph-tui/iterations/**) fail "staged forbidden: ${f}" ;;
      *.pid|*.log|*.parquet|*.sqlite|*.db|*.db-wal|*.db-shm|*.zip|*.tar|*.tgz|*.tar.gz) fail "staged forbidden artifact-like file: ${f}" ;;
    esac
  done <<<"${STAGED}"
fi
echo "[ok] staged set contains no forbidden paths"

# 3) Guardrail: prevent accidentally committing very large files.
# Use 10MB as "suspicious" threshold; tune if needed.
MAX_BYTES=$((10 * 1024 * 1024))
if [[ -n "${STAGED}" ]]; then
  LARGE_STAGED=0
  while IFS= read -r f; do
    if [[ -f "${f}" ]]; then
      sz="$(stat -c '%s' "${f}" 2>/dev/null || echo 0)"
      if [[ "${sz}" -gt "${MAX_BYTES}" ]]; then
        echo "ERROR: staged file >10MB: ${sz} bytes: ${f}" >&2
        LARGE_STAGED=1
      fi
    fi
  done <<<"${STAGED}"
  if [[ "${LARGE_STAGED}" == "1" ]]; then
    exit 1
  fi
fi
echo "[ok] no staged files >10MB"

# 4) SYNC_UP=1 preflight: untracked files in key paths won't be synced to VPS.
if [[ "${SYNC_UP_MODE}" == "1" ]]; then
  echo "[check] --sync-up: verify no untracked files in key sync paths"
  # Keep this list conservative: these are the paths most likely to be edited for a run.
  KEY_PATHS=(
    "docs"
    "coint4/configs"
    "coint4/artifacts/wfa/aggregate"
    "tasks"
  )
  UNTRACKED="$(git ls-files --others --exclude-standard -- "${KEY_PATHS[@]}" | sed -n '1,200p' || true)"
  if [[ -n "${UNTRACKED}" ]]; then
    echo "ERROR: untracked files exist under key paths (SYNC_UP=1 will NOT sync them):" >&2
    printf '%s\n' "${UNTRACKED}" >&2
    echo "Hint: stage/commit them (or delete) before running remote with SYNC_UP=1." >&2
    exit 1
  fi
  echo "[ok] no untracked files under key paths"
fi

echo "[done] verify_clean_cycle: OK"

