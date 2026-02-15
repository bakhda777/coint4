#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${repo_root}" ]]; then
  echo "ERROR: not inside a git repo" >&2
  exit 2
fi

if [[ "${PWD}" != "${repo_root}" ]]; then
  echo "ERROR: run from repo root: ${repo_root}" >&2
  exit 2
fi

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

echo "[preflight] repo_root=${repo_root}"

# Invariants: canonical structure
[[ -d "${repo_root}/coint4" ]] || fail "missing app-root dir: coint4/"
[[ -d "${repo_root}/docs" ]] || fail "missing docs/ at repo root"

if [[ ! -L "${repo_root}/coint4/docs" ]]; then
  fail "expected symlink: coint4/docs -> ../docs"
fi

docs_target="$(readlink "${repo_root}/coint4/docs")"
[[ "${docs_target}" == "../docs" ]] || fail "unexpected coint4/docs link target: ${docs_target} (expected ../docs)"

# Ralph TUI config + PRD JSON format sanity check
[[ -f "${repo_root}/.ralph-tui/config.toml" ]] || fail "missing .ralph-tui/config.toml"
[[ -f "${repo_root}/.ralph-tui/prd.json" ]] || fail "missing .ralph-tui/prd.json"

py_bin=""
if command -v python3 >/dev/null 2>&1; then
  py_bin="python3"
elif command -v python >/dev/null 2>&1; then
  py_bin="python"
else
  fail "python3 (or python) not found in PATH"
fi

"${py_bin}" - <<'PY'
import json
from pathlib import Path

path = Path(".ralph-tui/prd.json")
data = json.loads(path.read_text(encoding="utf-8"))

required_top = ["name", "branchName", "userStories"]
for k in required_top:
    if k not in data:
        raise SystemExit(f"prd.json missing top-level key: {k}")

if not isinstance(data["userStories"], list):
    raise SystemExit("prd.json userStories must be an array")

required_us = ["id", "title", "description", "acceptanceCriteria", "priority", "passes", "dependsOn"]
for i, us in enumerate(data["userStories"]):
    if not isinstance(us, dict):
        raise SystemExit(f"userStories[{i}] must be an object")
    missing = [k for k in required_us if k not in us]
    if missing:
        raise SystemExit(f"userStories[{i}] missing keys: {', '.join(missing)}")
    if not isinstance(us["acceptanceCriteria"], list):
        raise SystemExit(f"userStories[{i}].acceptanceCriteria must be an array")
    if not isinstance(us["dependsOn"], list):
        raise SystemExit(f"userStories[{i}].dependsOn must be an array")
print("[preflight] .ralph-tui/prd.json format: OK")
PY

echo "[preflight] make ci"
make ci

echo "[preflight] git status (porcelain)"
git status --porcelain=v1 || true

echo "[preflight] OK"
