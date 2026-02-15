#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command...>" >&2
  echo "Example: $0 bash scripts/optimization/watch_wfa_queue.sh --queue artifacts/wfa/aggregate/20260116_signal_grid/run_queue.csv" >&2
  exit 1
fi

API_BASE=${SERVSPACE_API_BASE:-"https://api.serverspace.ru/api/v1"}
API_KEY=${SERVSPACE_API_KEY:-""}
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

# Local-only convenience: if SERVSPACE_API_KEY isn't exported, read it from the gitignored secret file.
# Never echo this value.
if [[ -z "${API_KEY}" && -f "${LOCAL_REPO_DIR}/.secrets/serverspace_api_key" ]]; then
  API_KEY="$(tr -d '\n' < "${LOCAL_REPO_DIR}/.secrets/serverspace_api_key")"
fi

UPDATE_CODE=${UPDATE_CODE:-"1"}
SYNC_BACK=${SYNC_BACK:-"1"}
SYNC_PATHS=${SYNC_PATHS:-"docs coint4/artifacts coint4/results coint4/outputs"}
# If local repo is ahead of origin or git push isn't available, sync tracked files up to VPS.
SYNC_UP=${SYNC_UP:-"0"}
STOP_AFTER=${STOP_AFTER:-"1"}
# If SERVSPACE_API_KEY isn't available, you can still shut down via SSH after syncing back.
STOP_VIA_SSH=${STOP_VIA_SSH:-"0"}
SKIP_POWER=${SKIP_POWER:-"0"}

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
    echo "SERVSPACE_API_KEY is required to resolve server id (set env var or create .secrets/serverspace_api_key)." >&2
    return 1
  fi

  local target_name target_ip
  target_name="${SERVER_NAME}"
  target_ip="${SERVER_IP}"

  if [[ -z "$target_name" && -z "$target_ip" ]]; then
    echo "SERVER_ID, SERVER_NAME, or SERVER_IP is required (set env var)." >&2
    return 1
  fi

  api_get "servers" | python3 - "$target_name" "$target_ip" <<'PY'
import json
import sys

target_name = sys.argv[1].strip() or None
target_ip = sys.argv[2].strip() or None

IP_KEYS = ("ip", "public_ip", "ipv4", "address", "ip_address")
NESTED_KEYS = ("network", "networks", "addresses", "interfaces")


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
    data = json.load(sys.stdin)
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
    echo "SERVSPACE_API_KEY is required to start/stop server (set env var)." >&2
    exit 1
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
  echo "Set SERVSPACE_API_KEY (and leave SKIP_POWER=0) OR set STOP_VIA_SSH=1." >&2
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
  echo "[server] syncing tracked files from ${LOCAL_REPO_DIR} -> ${SERVER_USER}@${SERVER_IP}:${SERVER_REPO_DIR}"

  ssh "${SSH_OPTS[@]}" "${SERVER_USER}@${SERVER_IP}" "mkdir -p '${SERVER_REPO_DIR}'"
  git -C "$LOCAL_REPO_DIR" ls-files -z | rsync -az --from0 --files-from=- -e "ssh ${SSH_OPTS[*]}" "${LOCAL_REPO_DIR}/" "${SERVER_USER}@${SERVER_IP}:${SERVER_REPO_DIR}/"

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
