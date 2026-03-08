#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
STATE_DIR="$ROOT_DIR/artifacts/wfa/aggregate/.autonomous"
LOG_FILE="$STATE_DIR/watchdog.log"
WATCHDOG_STATE_FILE="$STATE_DIR/watchdog_state.json"
DRIVER_SCRIPT="$ROOT_DIR/scripts/optimization/autonomous_wfa_driver.sh"
SERVER_IP="${SERVER_IP:-85.198.90.128}"
SERVER_USER="${SERVER_USER:-root}"
WATCHDOG_MAX_IDLE_CYCLES="${WATCHDOG_MAX_IDLE_CYCLES:-1}"

mkdir -p "$STATE_DIR"

log() {
  printf '%s | %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*" >>"$LOG_FILE"
}

is_driver_running() {
  pgrep -f "scripts/optimization/autonomous_wfa_driver.sh" >/dev/null 2>&1
}

is_local_runner_busy() {
  python3 - <<'PY'
import os
import sys

patterns = (
    "run_wfa_queue_powered.py --queue",
    "watch_wfa_queue.sh --queue",
    "scripts/optimization/run_wfa_queue.py --queue",
)
self_pids = {str(os.getpid()), str(os.getppid())}

for pid in os.listdir("/proc"):
    if not pid.isdigit() or pid in self_pids:
        continue
    try:
        cmd = (
            open(f"/proc/{pid}/cmdline", "rb")
            .read()
            .replace(b"\x00", b" ")
            .decode("utf-8", "ignore")
            .strip()
        )
    except Exception:
        continue
    if not cmd:
        continue
    if "pgrep -f" in cmd or "python3 - <<" in cmd:
        continue
    if any(p in cmd for p in patterns):
        raise SystemExit(0)

raise SystemExit(1)
PY
}

remote_runner_count() {
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8 "$SERVER_USER@$SERVER_IP" "python3 - <<'PY'
import os

patterns = (
    'watch_wfa_queue.sh',
    'run_wfa_queue.py',
    'run_wfa_fullcpu.sh',
    'walk_forward',
    'postprocess_queue.py',
)

count = 0
for pid in os.listdir('/proc'):
    if not pid.isdigit():
        continue
    try:
        cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\\x00', b' ').decode('utf-8', 'ignore').strip()
    except Exception:
        continue
    if not cmd:
        continue
    if 'python3 - <<' in cmd or 'pgrep -f' in cmd:
        continue
    if any(p in cmd for p in patterns):
        count += 1

print(count)
PY" 2>/dev/null || echo "0"
}

queue_snapshot() {
  python3 - "$ROOT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
queue_root = root / "artifacts/wfa/aggregate"
best = None
for p in sorted(queue_root.rglob("run_queue.csv")):
    try:
        rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
    except Exception:
        continue
    planned = running = stalled = failed = completed = 0
    for r in rows:
        s = (r.get("status") or "").strip().lower()
        if s == "planned":
            planned += 1
        elif s == "running":
            running += 1
        elif s == "stalled":
            stalled += 1
        elif s in {"failed", "error"}:
            failed += 1
        elif s == "completed":
            completed += 1
    pending = planned + running + stalled + failed
    if pending <= 0:
        continue
    item = {
        "queue_rel": str(p.relative_to(root)),
        "pending": pending,
        "planned": planned,
        "running": running,
        "stalled": stalled,
        "failed": failed,
        "completed": completed,
        "total": len(rows),
    }
    if best is None:
        best = item
    else:
        if item["pending"] > best["pending"]:
            best = item
        elif item["pending"] == best["pending"] and item["stalled"] > best["stalled"]:
            best = item
if best is None:
    best = {"queue_rel": "", "pending": 0, "planned": 0, "running": 0, "stalled": 0, "failed": 0, "completed": 0, "total": 0}
print(json.dumps(best, ensure_ascii=False))
PY
}

start_driver() {
  local stamp
  stamp="$(date -u +%Y%m%d_%H%M%S)"
  local dlog="$STATE_DIR/driver_${stamp}.log"
  nohup "$DRIVER_SCRIPT" >>"$dlog" 2>&1 &
  log "WATCHDOG_DRIVER_START log=$dlog"
}

watchdog_state_get() {
  python3 - "$WATCHDOG_STATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    print('{"queue_rel":"","idle_cycles":0}')
    raise SystemExit(0)
try:
    data = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    data = {}
if not isinstance(data, dict):
    data = {}
print(json.dumps({"queue_rel": str(data.get("queue_rel", "")), "idle_cycles": int(data.get("idle_cycles", 0) or 0)}, ensure_ascii=False))
PY
}

watchdog_state_set() {
  local queue_rel="$1"
  local idle_cycles="$2"
  python3 - "$WATCHDOG_STATE_FILE" "$queue_rel" "$idle_cycles" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
queue_rel = sys.argv[2]
idle_cycles = int(float(sys.argv[3])) if sys.argv[3] else 0
state = {
    "queue_rel": queue_rel,
    "idle_cycles": max(0, idle_cycles),
}
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

trigger_powered_repair() {
  local queue_rel="$1"
  log "WATCHDOG_LIVENESS_ONLY queue=$queue_rel action=skip_powered_repair"
}

main() {
  local snap
  snap="$(queue_snapshot)"
  local queue_rel pending
  queue_rel="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d.get("queue_rel",""))' "$snap")"
  pending="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(int(d.get("pending",0)))' "$snap")"

  if ! is_driver_running; then
    if (( pending > 0 )); then
      log "WATCHDOG_DRIVER_DOWN pending=$pending queue=$queue_rel"
      start_driver
    else
      log "WATCHDOG_IDLE driver_down pending=0"
    fi
    watchdog_state_set "$queue_rel" 0
    return 0
  fi

  if (( pending <= 0 )); then
    watchdog_state_set "" 0
    log "WATCHDOG_OK pending=0"
    return 0
  fi

  if is_local_runner_busy; then
    watchdog_state_set "$queue_rel" 0
    log "WATCHDOG_OK local_runner_busy queue=$queue_rel pending=$pending"
    return 0
  fi

  local remote_count
  remote_count="$(remote_runner_count)"
  if [[ -z "$remote_count" || ! "$remote_count" =~ ^[0-9]+$ ]]; then
    remote_count=0
  fi

  if (( remote_count > 0 )); then
    watchdog_state_set "$queue_rel" 0
    log "WATCHDOG_OK remote_runner_count=$remote_count queue=$queue_rel pending=$pending"
    return 0
  fi

  local prev
  prev="$(watchdog_state_get)"
  local prev_queue prev_idle
  prev_queue="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d.get("queue_rel",""))' "$prev")"
  prev_idle="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(int(d.get("idle_cycles",0)))' "$prev")"

  local idle_cycles=1
  if [[ "$prev_queue" == "$queue_rel" ]]; then
    idle_cycles=$((prev_idle + 1))
  fi

  watchdog_state_set "$queue_rel" "$idle_cycles"
  log "WATCHDOG_IDLE_PENDING queue=$queue_rel pending=$pending idle_cycles=$idle_cycles remote_runner_count=0"

  if (( idle_cycles >= WATCHDOG_MAX_IDLE_CYCLES )); then
    trigger_powered_repair "$queue_rel"
    watchdog_state_set "$queue_rel" 0
  fi
}

main "$@"
