#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRD_REL="tasks/06/prd_optimization_loop_06.ralph.json"
PRD_PATH="${REPO_ROOT}/${PRD_REL}"

cd "${REPO_ROOT}"

command -v ralph-tui >/dev/null 2>&1 || { echo "ERROR: ralph-tui not found in PATH" >&2; exit 2; }
command -v codex >/dev/null 2>&1 || { echo "ERROR: codex not found in PATH" >&2; exit 2; }
[[ -f "${PRD_PATH}" ]] || { echo "ERROR: PRD not found: ${PRD_PATH}" >&2; exit 2; }

# Ensure runtime dirs exist and are writable for current user.
mkdir -p .ralph-tui/iterations .ralph-tui/worktrees
if [[ ! -w .ralph-tui ]]; then
  echo "ERROR: .ralph-tui is not writable by $(id -un). Fix ownership/permissions first." >&2
  exit 2
fi
if [[ ! -w tasks/06 ]]; then
  echo "ERROR: tasks/06 is not writable by $(id -un). Fix ownership/permissions first." >&2
  exit 2
fi

echo "[ralph-06] repo=${REPO_ROOT}"
echo "[ralph-06] prd=${PRD_REL}"
echo "[ralph-06] mode=no-sandbox"

exec ralph-tui run \
  --no-sandbox \
  --agent codex \
  --tracker json \
  --prd "${PRD_REL}"
