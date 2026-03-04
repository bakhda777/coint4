#!/usr/bin/env bash
set -euo pipefail

# Minute status snapshot for coint4 remote heavy runs.
# Prints a compact 1-line status suitable for chat delivery.
#
# Env:
#   SERVSPACE_API_BASE (optional) - default https://api.serverspace.ru/api/v1
#   SERVSPACE_API_KEY  (optional) - Serverspace API key (otherwise read from common key files)
#   SERVER_IP          (optional) - default 85.198.90.128
#   SERVER_ID          (optional) - Serverspace server id (otherwise resolved by IP)
#   SERVER_USER        (optional) - default root
#   SSH_KEY            (optional) - default ~/.ssh/id_ed25519
#   QUEUE              (optional) - queue path on VPS (relative to /opt/coint4/coint4)

API_BASE=${SERVSPACE_API_BASE:-"https://api.serverspace.ru/api/v1"}
SERVER_IP=${SERVER_IP:-"85.198.90.128"}
SERVER_ID=${SERVER_ID:-""}
SERVER_USER=${SERVER_USER:-"root"}
SSH_KEY=${SSH_KEY:-"${HOME}/.ssh/id_ed25519"}
QUEUE=${QUEUE:-""}

now_utc() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }

read_api_key() {
  if [[ -n "${SERVSPACE_API_KEY:-}" ]]; then
    printf '%s' "${SERVSPACE_API_KEY}"; return 0
  fi
  if [[ -n "${SERVERSPACE_API_KEY:-}" ]]; then
    printf '%s' "${SERVERSPACE_API_KEY}"; return 0
  fi
  for p in "${HOME}/.serverspace_api_key" "/etc/serverspace_api_key" "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/.secrets/serverspace_api_key"; do
    if [[ -f "$p" && -s "$p" ]]; then
      tr -d '\r\n' <"$p"; return 0
    fi
  done
  return 1
}

api_key=""
if api_key="$(read_api_key 2>/dev/null)"; then
  :
else
  api_key=""
fi

api_get() {
  local path="$1"
  curl -sS --max-time 5 -H 'content-type: application/json' -H "x-api-key: ${api_key}" "${API_BASE}/${path}" 2>/dev/null || true
}

resolve_server_id_by_ip() {
  local json
  json="$(api_get 'servers')"
  [[ -n "$json" ]] || return 1

  # NOTE: Don't combine a pipe into `python3 -` with a heredoc: the heredoc owns stdin.
  python3 -c 'import json,sys
ip=sys.argv[1]
data=json.load(sys.stdin)
items=data.get("data") or data.get("items") or data.get("servers") or data
if isinstance(items, dict):
    items=list(items.values())

def collect_ips(srv):
    ips=[]
    for k in ("ip","public_ip","ipv4","address","ip_address"):
        v=srv.get(k)
        if isinstance(v,str):
            ips.append(v)
        elif isinstance(v,list):
            ips += [x for x in v if isinstance(x,str)]
    for k in ("nics","interfaces","networks","network"):
        v=srv.get(k)
        if isinstance(v, dict):
            v=list(v.values())
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    vv=item.get("ip_address") or item.get("ip") or item.get("address")
                    if isinstance(vv,str):
                        ips.append(vv)
    return set([x for x in ips if x])

def srv_id(srv):
    for k in ("id","uuid","server_id"):
        v=srv.get(k)
        if v is not None and str(v):
            return str(v)
    return None

if isinstance(items, list):
    for srv in items:
        if isinstance(srv, dict) and ip in collect_ips(srv):
            sid=srv_id(srv)
            if sid:
                print(sid)
                raise SystemExit(0)
' "$SERVER_IP" <<<"$json" 2>/dev/null || true
}

power_state="n/a"
server_state="n/a"
if [[ -n "$api_key" ]]; then
  if [[ -z "$SERVER_ID" ]]; then
    SERVER_ID="$(resolve_server_id_by_ip || true)"
  fi
  if [[ -n "$SERVER_ID" ]]; then
    srv_json="$(api_get "servers/${SERVER_ID}")"
    if [[ -n "$srv_json" ]]; then
      power_state="$(python3 - <<'PY' 2>/dev/null || true
import json,sys
j=json.load(sys.stdin)
s=j.get('server') or j
v=s.get('is_power_on')
print('on' if v else 'off')
PY
<<<"$srv_json")"
      server_state="$(python3 - <<'PY' 2>/dev/null || true
import json,sys
j=json.load(sys.stdin)
s=j.get('server') or j
print(s.get('state') or s.get('status') or '')
PY
<<<"$srv_json")"
      power_state=${power_state:-unknown}
      server_state=${server_state:-unknown}
    fi
  fi
fi

ssh_ok=0
host="?"
up="?"
load="?"
queue_line="queue n/a"

SSH_OPTS=(-i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5)

if ssh "${SSH_OPTS[@]}" "${SERVER_USER}@${SERVER_IP}" 'echo ok' >/dev/null 2>&1; then
  ssh_ok=1
  host="$(ssh "${SSH_OPTS[@]}" "${SERVER_USER}@${SERVER_IP}" 'hostname' 2>/dev/null || true)"
  up="$(ssh "${SSH_OPTS[@]}" "${SERVER_USER}@${SERVER_IP}" "uptime -p | sed 's/^up //'" 2>/dev/null || true)"
  load="$(ssh "${SSH_OPTS[@]}" "${SERVER_USER}@${SERVER_IP}" "cut -d' ' -f1-3 /proc/loadavg" 2>/dev/null || true)"

  if [[ -n "$QUEUE" ]]; then
    queue_line="$(ssh "${SSH_OPTS[@]}" "${SERVER_USER}@${SERVER_IP}" \
      "cd /opt/coint4/coint4 && python3 - <<'PY'\nimport csv,sys\nfrom collections import Counter\npath=sys.argv[1]\ntry:\n  with open(path,newline='') as f:\n    st=[(row.get('status') or '').strip().lower() for row in csv.DictReader(f)]\n  c=Counter(st)\n  total=len(st)\n  planned=c.get('planned',0)\n  running=c.get('running',0)\n  completed=c.get('completed',0)\n  failed=c.get('failed',0)+c.get('error',0)\n  print(f'queue {path}: total={total} planned={planned} running={running} completed={completed} failed={failed}')\nexcept FileNotFoundError:\n  print(f'queue {path}: missing')\nPY" "$QUEUE" 2>/dev/null || true)"
  fi
fi

printf '%s | vps=%s/%s (%s) | ssh=%s | host=%s | up=%s | load=%s | %s\n' \
  "$(now_utc)" \
  "${power_state}" "${server_state}" "${SERVER_IP}" \
  "$ssh_ok" "${host:-?}" "${up:-?}" "${load:-?}" \
  "${queue_line}"
