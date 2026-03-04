#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QUEUE_ROOT="$ROOT_DIR/artifacts/wfa/aggregate"
STATE_DIR="$QUEUE_ROOT/.autonomous"
STATE_FILE="$STATE_DIR/driver_state.txt"
DECISION_NOTES_FILE="$STATE_DIR/decision_notes.jsonl"
FULLSPAN_CYCLE_STATE_FILE="$STATE_DIR/mini_cycle_state.txt"
LOG_FILE="$STATE_DIR/driver.log"
CANDIDATE_FILE="$STATE_DIR/candidate.csv"
HEARTBEAT_FILE="$STATE_DIR/heartbeat_state.json"
ORPHAN_FILE="$STATE_DIR/orphan_queues.csv"
SERVER_IP="${SERVER_IP:-85.198.90.128}"
SERVER_USER="${SERVER_USER:-root}"

qline="$(tr -d '\r\n' <"$STATE_FILE" 2>/dev/null || true)"
current_queue=""
if [[ -n "$qline" ]]; then
  current_queue="${qline##*queue=}"
  current_queue="${current_queue%% *}"
fi

candidate_top=""
if [[ -f "$CANDIDATE_FILE" ]]; then
  candidate_top="$(tail -n 1 "$CANDIDATE_FILE" 2>/dev/null || true)"
fi

heartbeat_for_current="none"

last_action='none'
if [[ -f "$LOG_FILE" ]]; then
  last_action="$(awk '/action=/{ if (match($0, /action=([^ ]+)/, a)) { print a[1]; } }' "$LOG_FILE" | tail -n 1 || true)"
fi
if [[ -z "$last_action" ]]; then
  last_action='none'
fi

last_fullspan_action="none"
if [[ -f "$DECISION_NOTES_FILE" ]]; then
  last_fullspan_action="$(python3 - "$DECISION_NOTES_FILE" -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); out="none";
for line in reversed(p.read_text(encoding=\'utf-8\').splitlines()):
    rec=json.loads(line)
    if rec.get(\'action\')==\'FULLSPAN_CYCLE\':
        out=f\"{rec.get(\'queue\', \'\')} {rec.get(\'next_step\', \'\')}\"; break
print(out)" )"
fi

if [[ -n "$current_queue" && -f "$HEARTBEAT_FILE" ]]; then
  heartbeat_for_current="$(python3 - "$current_queue" "$HEARTBEAT_FILE" <<'PY'
import json
import sys
from pathlib import Path

q = Path(sys.argv[1]).as_posix()
f = Path(sys.argv[2])
if not f.exists():
    raise SystemExit(0)

try:
    data = json.loads(f.read_text(encoding='utf-8'))
except Exception:
    raise SystemExit(0)

item = data.get(q)
if not item:
    item = data.get(('artifacts/' + q.lstrip('/')))
if not item:
    raise SystemExit(0)

print('pending={pending} completed={completed} running={running} stalled={stalled} rate_per_min={rate:.3f} eta_min={eta} stale_sec={stale:.1f} done={done}'.format(
    pending=item.get('pending'),
    completed=item.get('completed'),
    running=item.get('running'),
    stalled=item.get('stalled'),
    rate=float(item.get('rate_per_min', 0.0)),
    eta=item.get('eta_min'),
    stale=float(item.get('stale_sec', 0.0)),
    done=int(bool(item.get('done'))),
))
PY
)"
fi

orphans="0"
if [[ -f "$ORPHAN_FILE" ]]; then
  orphans="$(python3 - "$ORPHAN_FILE" <<'PY'
import csv
import sys
from pathlib import Path

p = Path(sys.argv[1])
try:
    with p.open(newline='', encoding='utf-8') as f:
        print(sum(1 for _ in csv.DictReader(f)))
except Exception:
    print(0)
PY
)"
fi

vps_processes="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=6 "$SERVER_USER@$SERVER_IP" 'ps aux | egrep -c "watch_wfa_queue|run_wfa_queue|run_wfa_fullcpu|python.*walk_forward" || true' 2>/dev/null || echo 0)"

active_queues="$(python3 - <<'PY'
import csv
from collections import Counter
from pathlib import Path
root = Path('/home/claudeuser/coint4/coint4/artifacts/wfa/aggregate')
out = []
for p in sorted(root.rglob('run_queue.csv')):
    try:
        rows = list(csv.DictReader(p.open(newline='')))
    except Exception:
        continue
    st = Counter((r.get('status') or '').strip().lower() for r in rows)
    pending = st.get('planned', 0) + st.get('running', 0) + st.get('stalled', 0) + st.get('failed', 0) + st.get('error', 0)
    if pending > 0:
        out.append((p.as_posix(), st.get('planned', 0), st.get('running', 0), st.get('stalled', 0), st.get('completed', 0), st.get('failed', 0) + st.get('error', 0)))
out = sorted(out, key=lambda x: -(x[1] + x[2] + x[3] + x[5]))
if not out:
    print('none')
else:
    for i, item in enumerate(out[:6], 1):
        pth, planned, running, stalled, completed, failed = item
        print(f"{i}) {pth} P={planned} R={running} S={stalled} F={failed} C={completed}")
PY
)"

printf '🧭 10m WFA report\n'
printf 'time=%s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
printf 'server=%s user=%s\n' "$SERVER_IP" "$SERVER_USER"
printf 'driver_state=%s\n' "${qline:-none}"
printf 'current=%s\n' "${current_queue:-none}"
printf 'candidate_top=%s\n' "${candidate_top:-none}"
printf 'heartbeat=%s\n' "$heartbeat_for_current"
printf 'last_action=%s\n' "$last_action"
printf 'orphan_count=%s\n' "$orphans"
printf 'vps_processes=%s\n' "$vps_processes"
printf 'active_queues=%s\n' "$active_queues"
printf 'recent_driver_log:\n'
if [[ -f "$LOG_FILE" ]]; then
  tail -n 8 "$LOG_FILE"
else
  echo 'missing'
fi

if [[ -f "$FULLSPAN_CYCLE_STATE_FILE" ]]; then
  cycle_count=$(wc -l < "$FULLSPAN_CYCLE_STATE_FILE" 2>/dev/null || echo 0)
else
  cycle_count=0
fi
printf "fullspan_cycle_state=%s\n" "$cycle_count"
printf "last_fullspan_action=%s\n" "$last_fullspan_action"
