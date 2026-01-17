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
LOCAL_REPO_DIR=${LOCAL_REPO_DIR:-"/root/coint4"}
UPDATE_CODE=${UPDATE_CODE:-"1"}
SYNC_BACK=${SYNC_BACK:-"1"}
SYNC_PATHS=${SYNC_PATHS:-"docs coint4/artifacts coint4/results coint4/outputs"}
STOP_AFTER=${STOP_AFTER:-"1"}
SKIP_POWER=${SKIP_POWER:-"0"}

SSH_OPTS=(
  -i "$SSH_KEY"
  -o BatchMode=yes
  -o StrictHostKeyChecking=no
  -o ConnectTimeout=5
)

api_get() {
  curl -sS -H 'content-type: application/json' -H "x-api-key: ${API_KEY}" "${API_BASE}/$1"
}

api_post() {
  curl -sS -X POST -H 'content-type: application/json' -H "x-api-key: ${API_KEY}" "${API_BASE}/$1" >/dev/null
}

resolve_server_id() {
  if [[ -n "$SERVER_ID" ]]; then
    echo "$SERVER_ID"
    return 0
  fi
  if [[ -z "$SERVER_NAME" ]]; then
    echo "SERVER_ID or SERVER_NAME is required (set env var)." >&2
    return 1
  fi
  if [[ -z "$API_KEY" ]]; then
    echo "SERVSPACE_API_KEY is required to resolve SERVER_NAME." >&2
    return 1
  fi
  api_get "servers" | python3 - "$SERVER_NAME" <<'PY'
import json
import sys

name = sys.argv[1]
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
    if srv.get("name") == name or srv.get("hostname") == name:
        for key in ("id", "uuid", "server_id"):
            if key in srv:
                print(srv[key])
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
  api_post "servers/${SERVER_ID}/start" || true
}

stop_server() {
  if [[ "$SKIP_POWER" == "1" || "$STOP_AFTER" != "1" ]]; then
    return 0
  fi
  if [[ -z "$API_KEY" ]]; then
    echo "SERVSPACE_API_KEY is required to stop server (set env var)." >&2
    exit 1
  fi
  if [[ -z "$SERVER_ID" ]]; then
    SERVER_ID=$(resolve_server_id)
  fi
  echo "[server] stopping ${SERVER_ID}"
  api_post "servers/${SERVER_ID}/stop" || true
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
  wait_for_ssh
  run_remote "$@"
  sync_back
  stop_server
}

main "$@"
