#!/usr/bin/env bash
set -euo pipefail

# Safe Ralph launcher:
# 1) preflight checks
# 2) attempt resume for target session id
# 3) fallback to fresh run with --no-sandbox

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  coint4/scripts/ralph/resume_or_restart.sh [session_id]

Default session_id: e4c4ded9

Behavior:
  1) Runs preflight checks (PRD/git/DNS/API)
  2) Tries `ralph-tui resume <session_id> --force`
  3) Falls back to `ralph-tui run --no-sandbox --prd .ralph-tui/prd.json`
EOF
  exit 0
fi

TARGET_INPUT="${1:-e4c4ded9}"
KNOWN_FULL_ID="e4c4ded9-ffe5-431b-b753-ff0309bb31b3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CANONICAL_PRD="${REPO_ROOT}/.ralph-tui/prd.json"
RESTART_PRD="${REPO_ROOT}/.ralph-tui/prd.restart.json"
TASKS_PRD="${REPO_ROOT}/tasks/prd.json"
RALPH_CONFIG="${REPO_ROOT}/.ralph-tui/config.toml"
SESSION_JSON="${REPO_ROOT}/.ralph-tui/session.json"
SESSION_META_JSON="${REPO_ROOT}/.ralph-tui/session-meta.json"
BACKUP_SESSION="${REPO_ROOT}/.ralph-tui.bak_20260219_124857/session.json"
REGISTRY_JSON="${HOME}/.config/ralph-tui/sessions.json"
CODEX_CONFIG="${HOME}/.codex/config.toml"
EFFECTIVE_PRD="${CANONICAL_PRD}"

if [[ "${TARGET_INPUT}" == "${KNOWN_FULL_ID}" ]]; then
  TARGET_FULL="${TARGET_INPUT}"
  TARGET_SHORT="${TARGET_INPUT:0:8}"
elif [[ "${TARGET_INPUT}" == "e4c4ded9" ]]; then
  TARGET_FULL="${KNOWN_FULL_ID}"
  TARGET_SHORT="e4c4ded9"
elif [[ "${TARGET_INPUT}" == *"-"* ]]; then
  TARGET_FULL="${TARGET_INPUT}"
  TARGET_SHORT="${TARGET_INPUT:0:8}"
else
  TARGET_FULL="${TARGET_INPUT}"
  TARGET_SHORT="${TARGET_INPUT:0:8}"
fi

log() {
  printf '[ralph-safe] %s\n' "$*"
}

warn() {
  printf '[ralph-safe][WARN] %s\n' "$*" >&2
}

die() {
  printf '[ralph-safe][ERROR] %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "command not found: $1"
}

fix_codex_collab_warning() {
  if [[ ! -f "${CODEX_CONFIG}" ]]; then
    return 0
  fi
  if ! rg -q '^[[:space:]]*collab[[:space:]]*=' "${CODEX_CONFIG}"; then
    return 0
  fi
  local tmp
  tmp="$(mktemp)"
  awk '!/^[[:space:]]*collab[[:space:]]*=/' "${CODEX_CONFIG}" > "${tmp}"
  mv "${tmp}" "${CODEX_CONFIG}"
  log "removed deprecated collab flag from ${CODEX_CONFIG}"
}

validate_prd_json() {
  [[ -f "${EFFECTIVE_PRD}" ]] || die "PRD not found: ${EFFECTIVE_PRD}"
  python3 - <<'PY' "${EFFECTIVE_PRD}" || exit 1
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
required = ("name", "branchName", "userStories")
missing = [k for k in required if k not in payload]
if missing:
    raise SystemExit(f"missing required PRD keys: {', '.join(missing)}")
if not isinstance(payload.get("userStories"), list):
    raise SystemExit("userStories must be array")
print("PRD_OK")
PY
}

prepare_effective_prd() {
  [[ -f "${CANONICAL_PRD}" ]] || die "canonical PRD not found: ${CANONICAL_PRD}"
  local open_tasks
  open_tasks="$(python3 - <<'PY' "${CANONICAL_PRD}"
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
payload = json.loads(p.read_text(encoding="utf-8"))
stories = payload.get("userStories")
if not isinstance(stories, list):
    stories = []
open_cnt = sum(1 for row in stories if isinstance(row, dict) and not bool(row.get("passes")))
print(open_cnt)
PY
)"

  if [[ "${open_tasks}" -gt 0 ]]; then
    EFFECTIVE_PRD="${CANONICAL_PRD}"
    return 0
  fi

  python3 - <<'PY' "${CANONICAL_PRD}" "${RESTART_PRD}"
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
payload = json.loads(src.read_text(encoding="utf-8"))
stories = payload.get("userStories")
if isinstance(stories, list):
    for row in stories:
        if isinstance(row, dict):
            row["passes"] = False
            if "completionNotes" in row and row["completionNotes"] is None:
                row.pop("completionNotes", None)
meta = payload.get("metadata")
if not isinstance(meta, dict):
    meta = {}
meta["updatedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
meta["sourcePrd"] = str(src)
meta["restartMode"] = "all_passes_reset_false"
payload["metadata"] = meta
dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(dst)
PY
  EFFECTIVE_PRD="${RESTART_PRD}"
  log "all stories already passed in canonical PRD; using restart PRD ${RESTART_PRD}"
}

check_prd_divergence() {
  if [[ ! -f "${TASKS_PRD}" ]]; then
    return 0
  fi
  local sum_a sum_b
  sum_a="$(sha256sum "${CANONICAL_PRD}" | awk '{print $1}')"
  sum_b="$(sha256sum "${TASKS_PRD}" | awk '{print $1}')"
  if [[ "${sum_a}" != "${sum_b}" ]]; then
    warn "PRD mismatch: using canonical ${CANONICAL_PRD}; tasks/prd.json differs"
  fi
}

check_git_writable() {
  [[ -d "${REPO_ROOT}/.git" ]] || die ".git directory not found at ${REPO_ROOT}"
  if [[ -e "${REPO_ROOT}/.git/index.lock" ]]; then
    die ".git/index.lock exists; remove stale lock before resume"
  fi
  [[ -w "${REPO_ROOT}/.git/index" ]] || die ".git/index is not writable"
}

check_remote_preflight() {
  require_cmd getent
  require_cmd curl
  getent hosts api.serverspace.ru >/dev/null || die "DNS failure: api.serverspace.ru"
  curl -sS --connect-timeout 8 --max-time 15 -o /dev/null "https://api.serverspace.ru" \
    || die "API unreachable: https://api.serverspace.ru"
  if [[ -z "${SERVSPACE_API_KEY:-}" && -z "${SERVERSPACE_API_KEY:-}" && ! -s "${HOME}/.serverspace_api_key" && ! -s "/etc/serverspace_api_key" && ! -s "${REPO_ROOT}/.secrets/serverspace_api_key" ]]; then
    warn "Serverspace API key is not set (set SERVSPACE_API_KEY/SERVERSPACE_API_KEY or create ~/.serverspace_api_key or /etc/serverspace_api_key or .secrets/serverspace_api_key)"
  fi
}

ensure_tracker_path() {
  [[ -f "${RALPH_CONFIG}" ]] || die "missing Ralph config: ${RALPH_CONFIG}"
  if ! rg -q '^[[:space:]]*path[[:space:]]*=[[:space:]]*".ralph-tui/prd.json"' "${RALPH_CONFIG}"; then
    warn "trackerOptions.path in ${RALPH_CONFIG} is not canonical (.ralph-tui/prd.json)"
  fi
}

restore_target_session_if_missing() {
  local listed
  listed="$(ralph-tui resume --list 2>&1 || true)"
  if rg -q "${TARGET_SHORT}" <<<"${listed}"; then
    return 0
  fi
  log "session ${TARGET_SHORT} is missing in registry; restoring resumable entry"
  python3 - <<'PY' "${REPO_ROOT}" "${TARGET_FULL}" "${EFFECTIVE_PRD}" "${BACKUP_SESSION}" "${SESSION_JSON}" "${SESSION_META_JSON}" "${REGISTRY_JSON}"
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

repo = Path(sys.argv[1])
target_id = sys.argv[2]
effective_prd = Path(sys.argv[3])
backup_session = Path(sys.argv[4])
session_json = Path(sys.argv[5])
session_meta_json = Path(sys.argv[6])
registry_json = Path(sys.argv[7])

now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

prd_path_rel = str(effective_prd)
try:
    prd_path_rel = str(effective_prd.relative_to(repo))
except ValueError:
    prd_path_rel = str(effective_prd)

if backup_session.exists():
    data = json.loads(backup_session.read_text(encoding="utf-8"))
else:
    prd_payload = json.loads(effective_prd.read_text(encoding="utf-8"))
    stories = prd_payload.get("userStories") if isinstance(prd_payload, dict) else []
    if not isinstance(stories, list):
        stories = []
    tasks = []
    for row in stories:
        if not isinstance(row, dict):
            continue
        tasks.append(
            {
                "id": str(row.get("id", "")).strip(),
                "title": str(row.get("title", "")).strip(),
                "status": "completed" if bool(row.get("passes")) else "open",
                "completedInSession": False,
            }
        )
    data = {
        "version": 1,
        "sessionId": target_id,
        "status": "interrupted",
        "startedAt": now,
        "updatedAt": now,
        "currentIteration": 0,
        "maxIterations": 0,
        "tasksCompleted": sum(1 for t in tasks if t["status"] == "completed"),
        "isPaused": False,
        "agentPlugin": "codex",
        "trackerState": {
            "plugin": "json",
            "prdPath": prd_path_rel,
            "totalTasks": len(tasks),
            "tasks": tasks,
        },
        "iterations": [],
        "skippedTaskIds": [],
        "cwd": str(repo),
        "activeTaskIds": [],
        "subagentPanelVisible": False,
    }

if not isinstance(data, dict):
    raise SystemExit("invalid backup session format")

data["sessionId"] = target_id
data["status"] = "interrupted"
data["updatedAt"] = now
data["cwd"] = str(repo)
data.setdefault("version", 1)
data.setdefault("currentIteration", 0)
data.setdefault("maxIterations", 0)
data.setdefault("tasksCompleted", 0)
data.setdefault("isPaused", False)
data.setdefault("agentPlugin", "codex")
data.setdefault("iterations", [])
data.setdefault("skippedTaskIds", [])
data.setdefault("activeTaskIds", [])
data.setdefault("subagentPanelVisible", False)

tracker = data.get("trackerState") if isinstance(data.get("trackerState"), dict) else {}
tracker["plugin"] = "json"
tracker["prdPath"] = prd_path_rel
tracker.setdefault("tasks", [])
if not isinstance(tracker["tasks"], list):
    tracker["tasks"] = []
tracker["totalTasks"] = len(tracker["tasks"])
data["trackerState"] = tracker

session_json.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

session_meta = {
    "id": target_id,
    "status": "interrupted",
    "startedAt": data.get("startedAt", now),
    "updatedAt": now,
    "agentPlugin": data.get("agentPlugin", "codex"),
    "trackerPlugin": "json",
    "currentIteration": int(data.get("currentIteration", 0) or 0),
    "maxIterations": int(data.get("maxIterations", 0) or 0),
    "totalTasks": int(tracker.get("totalTasks", 0) or 0),
    "tasksCompleted": int(data.get("tasksCompleted", 0) or 0),
    "cwd": str(repo),
}
session_meta_json.write_text(json.dumps(session_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

registry_json.parent.mkdir(parents=True, exist_ok=True)
if registry_json.exists():
    reg = json.loads(registry_json.read_text(encoding="utf-8"))
else:
    reg = {"version": 1, "sessions": {}}
if not isinstance(reg, dict):
    reg = {"version": 1, "sessions": {}}
reg.setdefault("version", 1)
reg.setdefault("sessions", {})
if not isinstance(reg["sessions"], dict):
    reg["sessions"] = {}
reg["sessions"][target_id] = {
    "sessionId": target_id,
    "cwd": str(repo),
    "status": "interrupted",
    "startedAt": data.get("startedAt", now),
    "updatedAt": now,
    "agentPlugin": data.get("agentPlugin", "codex"),
    "trackerPlugin": "json",
    "prdPath": str(effective_prd),
    "sandbox": False,
}
registry_json.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("SESSION_RESTORED", target_id)
PY
}

repoint_session_to_effective_prd() {
  python3 - <<'PY' "${REPO_ROOT}" "${TARGET_FULL}" "${EFFECTIVE_PRD}" "${SESSION_JSON}" "${SESSION_META_JSON}" "${REGISTRY_JSON}"
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

repo = Path(sys.argv[1])
target_id = sys.argv[2]
effective_prd = Path(sys.argv[3])
session_json = Path(sys.argv[4])
session_meta_json = Path(sys.argv[5])
registry_json = Path(sys.argv[6])

now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
try:
    prd_rel = str(effective_prd.relative_to(repo))
except ValueError:
    prd_rel = str(effective_prd)

if session_json.exists():
    data = json.loads(session_json.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        tracker = data.get("trackerState")
        if not isinstance(tracker, dict):
            tracker = {}
        tracker["plugin"] = "json"
        tracker["prdPath"] = prd_rel
        data["trackerState"] = tracker
        data["updatedAt"] = now
        if str(data.get("sessionId", "")) == target_id:
            data["status"] = "interrupted"
        session_json.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if session_meta_json.exists():
    meta = json.loads(session_meta_json.read_text(encoding="utf-8"))
    if isinstance(meta, dict):
        meta["updatedAt"] = now
        if str(meta.get("id", "")) == target_id:
            meta["status"] = "interrupted"
        session_meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if registry_json.exists():
    reg = json.loads(registry_json.read_text(encoding="utf-8"))
else:
    reg = {"version": 1, "sessions": {}}
if not isinstance(reg, dict):
    reg = {"version": 1, "sessions": {}}
reg.setdefault("sessions", {})
if isinstance(reg["sessions"], dict):
    entry = reg["sessions"].get(target_id)
    if isinstance(entry, dict):
        entry["prdPath"] = str(effective_prd)
        entry["status"] = "interrupted"
        entry["updatedAt"] = now
        reg["sessions"][target_id] = entry
        registry_json.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

main() {
  require_cmd ralph-tui
  require_cmd python3
  require_cmd rg

  cd "${REPO_ROOT}"

  fix_codex_collab_warning
  prepare_effective_prd
  ensure_tracker_path
  validate_prd_json
  check_prd_divergence
  check_git_writable
  check_remote_preflight
  restore_target_session_if_missing
  repoint_session_to_effective_prd

  log "trying: ralph-tui resume ${TARGET_SHORT} --force"
  if ralph-tui resume "${TARGET_SHORT}" --force; then
    exit 0
  fi

  warn "resume failed; starting fresh run with ${EFFECTIVE_PRD} and no sandbox"
  exec ralph-tui run --no-sandbox --prd "${EFFECTIVE_PRD}"
}

main "$@"
