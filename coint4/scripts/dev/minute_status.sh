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
#   SSH_HOSTS          (optional) - comma-separated SSH host aliases (fallbacks), default IP, coint
#
# Design notes:
# - Do not report SSH outage as a hard VPS error if API/remote can be reached via retries.
# - Prefer real SSH probe outcome over API state when they differ.
# - Keep output one-line, deterministic and short for cron delivery.

API_BASE=${SERVSPACE_API_BASE:-"https://api.serverspace.ru/api/v1"}
SERVER_IP=${SERVER_IP:-"85.198.90.128"}
SERVER_ID=${SERVER_ID:-""}
SERVER_USER=${SERVER_USER:-"root"}
SSH_KEY=${SSH_KEY:-"${HOME}/.ssh/id_ed25519"}
QUEUE=${QUEUE:-""}
DRIVER_STATE_FILE="${DRIVER_STATE_FILE:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/artifacts/wfa/aggregate/.autonomous/driver_state.txt"}"
SSH_HOSTS_RAW=${SSH_HOSTS:-"${SERVER_IP},coint"}

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

# Build SSH probe targets list
IFS=',' read -r -a SSH_TARGETS <<<"$SSH_HOSTS_RAW"

ssh_ok=0
ssh_host="?"
ssh_up="?"
ssh_load="?"
queue_line="queue n/a"
ssh_reason=""
last_stdout=""

SSH_OPTS=(-i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5)

probe_remote() {
  local target="$1"
  local rc=1

  # Single-shot command to reduce transient SSH failures and collect everything consistently.
  if out="$(ssh "${SSH_OPTS[@]}" "${SERVER_USER}@${target}" 'hostname; echo "UP=$(uptime -p 2>/dev/null | sed "s/^up //")"; echo "LOAD=$(cut -d" " -f1-3 /proc/loadavg 2>/dev/null)"' 2>&1)"; then
    rc=0
    last_stdout="$out"
  else
    rc=$?
    ssh_reason="${out//$'\n'/ }"
  fi

  if [[ $rc -eq 0 ]]; then
    ssh_ok=1
    ssh_host="$(printf '%s\n' "$out" | sed -n '1p' | tr -d '\r')"
    ssh_up="$(printf '%s\n' "$out" | sed -n '2p' | sed 's/^UP=//')"
    ssh_load="$(printf '%s\n' "$out" | sed -n '3p' | sed 's/^LOAD=//')"

    if [[ -n "$QUEUE" ]]; then
      queue_line="$(ssh "${SSH_OPTS[@]}" "${SERVER_USER}@${target}" "cd /opt/coint4/coint4 && python3 - <<'PY'
import csv
from collections import Counter
path='${QUEUE}'
try:
    with open(path,newline='') as f:
        rows=list(csv.DictReader(f))
except FileNotFoundError:
    print('queue ' + path + ': missing')
    raise SystemExit(0)
except Exception as e:
    print('queue ' + path + ': unreadable (' + str(e) + ')')
    raise SystemExit(0)

statuses=[(r.get('status') or '').strip().lower() for r in rows]
c=Counter(statuses)
print('queue %s: total=%d planned=%d running=%d completed=%d failed=%d' % (
    path,
    len(rows),
    c.get('planned', 0),
    c.get('running', 0),
    c.get('completed', 0),
    c.get('failed', 0) + c.get('error', 0),
))
PY" 2>/dev/null || true)"
    fi

    return 0
  fi

  return 1
}

probe_ok=0
for host_try in "${SSH_TARGETS[@]}"; do
  # trim
  host_try="${host_try// /}"
  [[ -z "$host_try" ]] && continue

  for attempt in 1 2 3; do
    if probe_remote "$host_try"; then
      probe_ok=1
      break 2
    fi
    sleep 1
  done
done

# If SSH is reachable, trust runtime reality for final state.
if [[ "$ssh_ok" -eq 1 ]]; then
  if [[ "$power_state" == "off" ]]; then
    power_state="on"
    server_state="${server_state:-online}"
  fi
else
  # If API says on but SSH fails, keep as an operational warning rather than a hard stop.
  if [[ "$power_state" == "on" ]]; then
    power_state="on"
  fi
fi

driver_state=""
if [[ -f "$DRIVER_STATE_FILE" ]]; then
  driver_state="$(tr -d '\r\n' < "$DRIVER_STATE_FILE")"
fi

printf '%s | vps=%s/%s (%s) | ssh=%s | host=%s | up=%s | load=%s | %s' \
  "$(now_utc)" \
  "${power_state}" "${server_state}" "${SERVER_IP}" \
  "${ssh_ok}" "${ssh_host}" "${ssh_up}" "${ssh_load}" "${queue_line}"
if [[ -n "$driver_state" ]]; then
  printf ' | driver=%s' "$driver_state"
fi
if [[ "$ssh_ok" -eq 0 && -n "$ssh_reason" ]]; then
  printf ' | ssh_err=%s' "$ssh_reason"
fi
printf '\n'
