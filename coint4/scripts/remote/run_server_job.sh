#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command...>" >&2
  echo "Example: $0 bash scripts/optimization/watch_wfa_queue.sh --queue artifacts/wfa/aggregate/20260116_signal_grid/run_queue.csv" >&2
  exit 1
fi

API_BASE=${SERVSPACE_API_BASE:-"https://api.serverspace.ru/api/v1"}
API_KEY=${SERVSPACE_API_KEY:-${SERVERSPACE_API_KEY:-""}}
SERVER_ID=${SERVER_ID:-""}
SERVER_NAME=${SERVER_NAME:-""}
SERVER_IP=${SERVER_IP:-"85.198.90.128"}
SERVER_USER=${SERVER_USER:-"root"}
SSH_KEY=${SSH_KEY:-"${HOME}/.ssh/id_ed25519"}
SERVER_REPO_DIR=${SERVER_REPO_DIR:-"/opt/coint4"}
SERVER_WORK_DIR=${SERVER_WORK_DIR:-"/opt/coint4/coint4"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LOCAL_REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOCAL_REPO_DIR=${LOCAL_REPO_DIR:-"${DEFAULT_LOCAL_REPO_DIR}"}

load_api_key_from_file() {
  local path="$1"
  if [[ -n "${API_KEY}" ]]; then
    return 0
  fi
  if [[ ! -f "${path}" ]]; then
    return 0
  fi
  if [[ ! -s "${path}" ]]; then
    echo "Serverspace API key file is empty: ${path}" >&2
    exit 2
  fi
  if ! API_KEY="$(tr -d '\n' < "${path}" | tr -d '\r')"; then
    echo "Cannot read Serverspace API key file: ${path}" >&2
    exit 2
  fi
}

# Prefer external key files to avoid relying on env vars.
# Never echo the key value.
load_api_key_from_file "${HOME}/.serverspace_api_key"
load_api_key_from_file "/etc/serverspace_api_key"
# Legacy fallback: repo-local gitignored secret file.
load_api_key_from_file "${LOCAL_REPO_DIR}/.secrets/serverspace_api_key"

UPDATE_CODE=${UPDATE_CODE:-"1"}
SYNC_BACK=${SYNC_BACK:-"1"}
SYNC_PATHS=${SYNC_PATHS:-"docs coint4/artifacts coint4/results coint4/outputs"}
# If local repo is ahead of origin or git push isn't available, sync tracked files up to VPS.
SYNC_UP=${SYNC_UP:-"0"}
# SYNC_UP_MODE=tracked  -> sync all git-tracked files.
# SYNC_UP_MODE=code     -> sync tracked files, excluding artifacts/outputs/logs/pids.
SYNC_UP_MODE=${SYNC_UP_MODE:-"tracked"}
STOP_AFTER=${STOP_AFTER:-"1"}
# If SERVSPACE_API_KEY isn't available, you can still shut down via SSH after syncing back.
STOP_VIA_SSH=${STOP_VIA_SSH:-"0"}
SKIP_POWER=${SKIP_POWER:-"0"}
ACTIVE_BATCH_STOP_GUARD=${ACTIVE_BATCH_STOP_GUARD:-"1"}
ALLOW_STOP_DURING_ACTIVE_BATCH=${ALLOW_STOP_DURING_ACTIVE_BATCH:-"0"}
ACTIVE_BATCH_GUARD_SEC=${ACTIVE_BATCH_GUARD_SEC:-"3600"}

# Optional preflight guard (idempotency): run once after SSH is ready, but before any update/sync/stop.
# If the command prints PREFLIGHT_MATCH and exits 0, this script exits 0 immediately (no side effects).
REMOTE_PREFLIGHT_CMD=${REMOTE_PREFLIGHT_CMD:-""}
PREFLIGHT_MATCH=${PREFLIGHT_MATCH:-""}

CLEANUP_ENABLED=0
DID_STOP=0

SSH_OPTS=(
  -i "$SSH_KEY"
  -o BatchMode=yes
  -o StrictHostKeyChecking=no
  -o ConnectTimeout=5
)

if [[ "$SYNC_UP" == "1" && "$UPDATE_CODE" == "1" ]]; then
  echo "[local] SYNC_UP=1 => forcing UPDATE_CODE=0 (avoid git pull conflicts)" >&2
  UPDATE_CODE="0"
fi

api_get() {
  curl -sS -H 'content-type: application/json' -H "x-api-key: ${API_KEY}" "${API_BASE}/$1"
}

api_post() {
  local path=$1
  local data=${2:-"{}"}
  curl -sS -X POST -H 'content-type: application/json' -H "x-api-key: ${API_KEY}" -d "${data}" "${API_BASE}/${path}" >/dev/null
}

resolve_server_id() {
  if [[ -n "$SERVER_ID" ]]; then
    echo "$SERVER_ID"
    return 0
  fi
  if [[ -z "$API_KEY" ]]; then
    echo "Serverspace API key is required to resolve server id (set SERVSPACE_API_KEY/SERVERSPACE_API_KEY or create ~/.serverspace_api_key or /etc/serverspace_api_key)." >&2
    return 2
  fi

  local target_name target_ip
  target_name="${SERVER_NAME}"
  target_ip="${SERVER_IP}"

  if [[ -z "$target_name" && -z "$target_ip" ]]; then
    echo "SERVER_ID, SERVER_NAME, or SERVER_IP is required (set env var)." >&2
    return 1
  fi

  # NOTE: Don't combine a pipe of JSON into `python3 -` with a heredoc of python code:
  # the heredoc owns stdin and the JSON never reaches python. Fetch JSON to a temp file instead.
  local tmp_servers_json
  tmp_servers_json="$(mktemp)"
  trap 'rm -f "$tmp_servers_json"' RETURN
  api_get "servers" >"$tmp_servers_json"
  python3 - "$target_name" "$target_ip" "$tmp_servers_json" <<'PY'
import json
import sys

target_name = sys.argv[1].strip() or None
target_ip = sys.argv[2].strip() or None
servers_path = sys.argv[3]

IP_KEYS = ("ip", "public_ip", "ipv4", "address", "ip_address")
# Serverspace API returns NICs under "nics" with "ip_address".
NESTED_KEYS = ("network", "networks", "addresses", "interfaces", "nics")


def _as_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, dict):
        return list(v.values())
    return []


def collect_ips(srv):
    ips = []
    for k in IP_KEYS:
        v = srv.get(k)
        if isinstance(v, str):
            ips.append(v)
        elif isinstance(v, list):
            ips.extend([x for x in v if isinstance(x, str)])

    for k in NESTED_KEYS:
        v = srv.get(k)
        for item in _as_list(v):
            if isinstance(item, str):
                ips.append(item)
                continue
            if not isinstance(item, dict):
                continue
            for kk in IP_KEYS:
                vv = item.get(kk)
                if isinstance(vv, str):
                    ips.append(vv)

    # Deduplicate while keeping order.
    out = []
    for ip in ips:
        if ip and ip not in out:
            out.append(ip)
    return out


def get_id(srv):
    for key in ("id", "uuid", "server_id"):
        v = srv.get(key)
        if isinstance(v, (str, int)) and str(v):
            return str(v)
    return None


def name_matches(srv, name):
    return srv.get("name") == name or srv.get("hostname") == name


def ip_matches(srv, ip):
    return ip in collect_ips(srv)


try:
    with open(servers_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
except Exception:
    sys.exit(1)

if isinstance(data, dict):
    data = data.get("data") or data.get("items") or data.get("servers") or data

if isinstance(data, dict):
    data = list(data.values())

if not isinstance(data, list):
    sys.exit(1)

for srv in data:
    if not isinstance(srv, dict):
        continue

    if target_name and name_matches(srv, target_name):
        sid = get_id(srv)
        if sid:
            print(sid)
            sys.exit(0)

    if target_ip and ip_matches(srv, target_ip):
        sid = get_id(srv)
        if sid:
            print(sid)
            sys.exit(0)
print("")
PY
}

start_server() {
  if [[ "$SKIP_POWER" == "1" ]]; then
    return 0
  fi
  if [[ -z "$API_KEY" ]]; then
    echo "Serverspace API key is required to start/stop server (set SERVSPACE_API_KEY/SERVERSPACE_API_KEY or create ~/.serverspace_api_key or /etc/serverspace_api_key)." >&2
    exit 2
  fi
  local sid
  sid=$(resolve_server_id)
  if [[ -z "$sid" ]]; then
    echo "Unable to resolve server id." >&2
    exit 1
  fi
  SERVER_ID="$sid"
  echo "[server] starting ${SERVER_ID}"
  api_post "servers/${SERVER_ID}/power/on" "{\"server_id\":\"${SERVER_ID}\"}" || true
}

stop_server() {
  if [[ "$STOP_AFTER" != "1" ]]; then
    return 0
  fi

  if [[ "$ACTIVE_BATCH_STOP_GUARD" == "1" && "$ALLOW_STOP_DURING_ACTIVE_BATCH" != "1" ]]; then
    local batch_state=""
    local guard_hit=""
    for candidate in \
      "$LOCAL_REPO_DIR/coint4/artifacts/wfa/aggregate/.autonomous/batch_session_state.json" \
      "$LOCAL_REPO_DIR/artifacts/wfa/aggregate/.autonomous/batch_session_state.json"; do
      if [[ -f "$candidate" ]]; then
        batch_state="$candidate"
        break
      fi
    done
    if [[ -n "$batch_state" ]]; then
      guard_hit="$(python3 - "$batch_state" "$ACTIVE_BATCH_GUARD_SEC" <<'PY'
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
guard_sec = max(0, int(float(sys.argv[2] or 0)))
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("0")
    raise SystemExit(0)

active = bool(payload.get("active"))
last_dispatch_epoch = int(float(payload.get("last_dispatch_epoch", 0) or 0))
start_epoch = int(float(payload.get("start_epoch", 0) or 0))
anchor = max(last_dispatch_epoch, start_epoch)
now = int(time.time())
recent = anchor > 0 and (now - anchor) <= guard_sec
print("1" if active and recent else "0")
PY
)"
      if [[ "$guard_hit" == "1" ]]; then
        echo "[server] stop skipped: active batch session guard (${batch_state})"
        return 0
      fi
    fi
  fi

  if [[ "$SKIP_POWER" != "1" && -n "$API_KEY" ]]; then
    if [[ -z "$SERVER_ID" ]]; then
      SERVER_ID=$(resolve_server_id)
    fi
    echo "[server] stopping ${SERVER_ID} (API)"
    api_post "servers/${SERVER_ID}/power/shutdown" "{\"server_id\":\"${SERVER_ID}\"}" || true
    return 0
  fi

  if [[ "$STOP_VIA_SSH" == "1" ]]; then
    echo "[server] stopping via SSH (shutdown -h now)"
    ssh "${SSH_OPTS[@]}" "${SERVER_USER}@${SERVER_IP}" "shutdown -h now" || true
    return 0
  fi

  echo "Unable to stop VPS automatically." >&2
  echo "Set SERVSPACE_API_KEY/SERVERSPACE_API_KEY (or ~/.serverspace_api_key or /etc/serverspace_api_key) OR set STOP_VIA_SSH=1." >&2
  return 1
}

wait_for_ssh() {
  echo "[server] waiting for SSH on ${SERVER_USER}@${SERVER_IP}"
  local start_ts
  start_ts=$(date +%s)
  while true; do
    if ssh "${SSH_OPTS[@]}" "${SERVER_USER}@${SERVER_IP}" "echo ok" >/dev/null 2>&1; then
      echo "[server] SSH ready"
      return 0
    fi
    sleep 5
    if [[ $(( $(date +%s) - start_ts )) -gt 900 ]]; then
      echo "SSH not ready after 15 minutes." >&2
      return 1
    fi
  done
}

cleanup() {
  local rc=$?
  if [[ "${CLEANUP_ENABLED}" != "1" ]]; then
    exit "${rc}"
  fi
  if [[ "${DID_STOP}" == "1" ]]; then
    exit "${rc}"
  fi

  # Best-effort shutdown to avoid leaving an expensive VPS running if anything fails mid-run.
  set +e
  stop_server
  set -e
  exit "${rc}"
}

trap cleanup EXIT

sync_up() {
  if [[ "$SYNC_UP" != "1" ]]; then
    return 0
  fi
  if [[ ! -d "$LOCAL_REPO_DIR" ]]; then
    echo "LOCAL_REPO_DIR not found: $LOCAL_REPO_DIR" >&2
    exit 1
  fi
  if [[ ! -d "$LOCAL_REPO_DIR/.git" ]]; then
    echo "LOCAL_REPO_DIR is not a git repo (missing .git): $LOCAL_REPO_DIR" >&2
    exit 1
  fi
  if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync not found (required for SYNC_UP=1)" >&2
    exit 1
  fi
  if ! command -v git >/dev/null 2>&1; then
    echo "git not found (required for SYNC_UP=1)" >&2
    exit 1
  fi

  local sha
  sha="$(git -C "$LOCAL_REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
  echo "[server] syncing files (${SYNC_UP_MODE}) from ${LOCAL_REPO_DIR} -> ${SERVER_USER}@${SERVER_IP}:${SERVER_REPO_DIR}"

  ssh "${SSH_OPTS[@]}" "${SERVER_USER}@${SERVER_IP}" "mkdir -p '${SERVER_REPO_DIR}'"
  if [[ "$SYNC_UP_MODE" == "tracked" ]]; then
    git -C "$LOCAL_REPO_DIR" ls-files -z | rsync -az --from0 --files-from=- -e "ssh ${SSH_OPTS[*]}" "${LOCAL_REPO_DIR}/" "${SERVER_USER}@${SERVER_IP}:${SERVER_REPO_DIR}/"
  elif [[ "$SYNC_UP_MODE" == "code" ]]; then
    git -C "$LOCAL_REPO_DIR" ls-files -z \
      | while IFS= read -r -d '' rel_path; do
          case "$rel_path" in
            coint4/artifacts/*|coint4/outputs/*|outputs/*|.ralph-tui/iterations/*|*.log|*.pid)
              continue
              ;;
          esac
          printf '%s\0' "$rel_path"
        done \
      | rsync -az --from0 --files-from=- -e "ssh ${SSH_OPTS[*]}" "${LOCAL_REPO_DIR}/" "${SERVER_USER}@${SERVER_IP}:${SERVER_REPO_DIR}/"
  else
    echo "Unsupported SYNC_UP_MODE=${SYNC_UP_MODE}. Use tracked or code." >&2
    exit 1
  fi

  if [[ -n "$sha" ]]; then
    ssh "${SSH_OPTS[@]}" "${SERVER_USER}@${SERVER_IP}" "echo '${sha}' > '${SERVER_REPO_DIR}/SYNCED_FROM_COMMIT.txt'"
  fi
}

shell_quote_single() {
  local s="$1"
  # Wrap a string in single quotes and escape embedded single quotes as: '\''.
  s=${s//\'/\'\\\'\'}
  printf "'%s'" "$s"
}

preflight_guard() {
  if [[ -z "${REMOTE_PREFLIGHT_CMD}" || -z "${PREFLIGHT_MATCH}" ]]; then
    return 0
  fi

  local out rc quoted
  quoted="$(shell_quote_single "${REMOTE_PREFLIGHT_CMD}")"

  # Remote preflight is allowed to fail (e.g. session absent => exit 1). We only early-exit on match.
  set +e
  out="$(ssh "${SSH_OPTS[@]}" "${SERVER_USER}@${SERVER_IP}" "bash -lc ${quoted}" 2>&1)"
  rc=$?
  set -e

  if [[ "${rc}" == "0" && "${out}" == *"${PREFLIGHT_MATCH}"* ]]; then
    # Print preflight output and exit cleanly without any update/sync/stop/cleanup.
    printf '%s\n' "${out}"
    CLEANUP_ENABLED=0
    DID_STOP=1
    exit 0
  fi
}

run_remote() {
  local cmd
  cmd=$(printf '%q ' "$@")
  if [[ "$UPDATE_CODE" == "1" ]]; then
    ssh "${SSH_OPTS[@]}" "${SERVER_USER}@${SERVER_IP}" "cd '${SERVER_REPO_DIR}' && git pull && cd '${SERVER_WORK_DIR}' && ${cmd}"
  else
    ssh "${SSH_OPTS[@]}" "${SERVER_USER}@${SERVER_IP}" "cd '${SERVER_WORK_DIR}' && ${cmd}"
  fi
}

sync_back() {
  if [[ "$SYNC_BACK" != "1" ]]; then
    return 0
  fi
  for path in $SYNC_PATHS; do
    local src
    local dst
    src="${SERVER_REPO_DIR}/${path}/"
    dst="${LOCAL_REPO_DIR}/${path}/"
    mkdir -p "$dst"
    rsync -az -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no" "${SERVER_USER}@${SERVER_IP}:${src}" "$dst"
  done
}

main() {
  start_server
  CLEANUP_ENABLED=1
  wait_for_ssh
  preflight_guard
  sync_up
  run_remote "$@"
  sync_back
  DID_STOP=1
  stop_server
}

main "$@"
