#!/usr/bin/env bash
set -euo pipefail

# Lightweight orchestrator: keeps advancing through non-terminal run_queues.
# It picks the next queue with pending work and starts run_wfa_queue_powered
# only when no other WFA queue runner is active.

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
QUEUE_ROOT="$ROOT_DIR/artifacts/wfa/aggregate"
STATE_DIR="$ROOT_DIR/artifacts/wfa/aggregate/.autonomous"
STATE_FILE="$STATE_DIR/driver_state.txt"
LOG_FILE="$STATE_DIR/driver.log"
SERVER_IP="${SERVER_IP:-85.198.90.128}"
SERVER_USER="${SERVER_USER:-root}"
LOCK_FILE="$STATE_DIR/driver.lock"
CANDIDATE_FILE="$STATE_DIR/candidate.csv"

mkdir -p "$STATE_DIR"

log() {
  printf '%s | %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*" >> "$LOG_FILE"
}

# Keep single driver instance.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "driver_already_running"
  exit 0
fi
trap 'flock -u 9; rm -f "$LOCK_FILE"' EXIT

find_candidate() {
  python3 - <<'PY' "$QUEUE_ROOT" "$CANDIDATE_FILE"
import csv
import os
import sys
from collections import Counter
from pathlib import Path

queue_root = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
out = []

for p in sorted(queue_root.rglob('run_queue.csv')):
    try:
        rows = list(csv.DictReader(p.open(newline='')))
    except Exception:
        continue

    st = Counter((r.get('status') or '').strip().lower() for r in rows)
    planned = int(st.get('planned', 0))
    running = int(st.get('running', 0))
    stalled = int(st.get('stalled', 0))
    failed = int(st.get('failed', 0)) + int(st.get('error', 0))
    total = len(rows)

    if total == 0:
        continue

    pending = planned + running + stalled + failed
    if pending == 0:
        continue

    # Priority: active > stalled > planned.
    if running > 0:
        priority = 1
    elif stalled > 0:
        priority = 2
    else:
        priority = 3

    # Prefer older queues at equal priority.
    mtime = int(p.stat().st_mtime)
    out.append((priority, mtime, str(p), planned, running, stalled, failed, st.get('completed', 0), total))

out.sort(key=lambda x: (x[0], x[1]))

with out_csv.open('w', encoding='utf-8', newline='') as f:
    if not out:
        f.write('queue,planned,running,stalled,failed,completed,total,priority\n')
        sys.exit(0)
    r = out[0]
    f.write('queue,planned,running,stalled,failed,completed,total,priority\n')
    f.write('%s,%d,%d,%d,%d,%s,%d,%d\n' % (r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[0]))
PY
}

_is_match_running() {
  local pattern="$1"
  local pids
  pids="$(pgrep -f -- "$pattern" || true)"
  if [[ -z "$pids" ]]; then
    return 1
  fi

  while IFS= read -r pid; do
    if [[ -z "$pid" ]]; then
      continue
    fi
    if [[ "$pid" != "$$" ]]; then
      return 0
    fi
  done <<< "$pids"

  return 1
}


stale_running() {
  local queue_rel="$1"
  local queue_abs="$ROOT_DIR/$queue_rel"

  if [[ -f "$queue_abs" ]]; then
    python3 - <<'PY2' "$queue_abs"
import csv
from pathlib import Path
import sys

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open(newline="")))
changed = 0
for row in rows:
    if (row.get("status") or "").strip().lower() == "running":
        row["status"] = "stalled"
        changed += 1
if changed > 0 and rows:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
PY2
  fi

  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=6 "$SERVER_USER@$SERVER_IP" "python3 - <<'PY2'
import csv
from pathlib import Path
import os

queue_path = Path('/opt/coint4/coint4') / '$queue_rel'
rows = list(csv.DictReader(queue_path.open(newline='')))
changed = 0
for row in rows:
    if (row.get('status') or '').strip().lower() == 'running':
        row['status'] = 'stalled'
        changed += 1
if changed > 0 and rows:
    with queue_path.open('w', newline='') as f:
        import csv as _c
        w = _c.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print('stale_mark_running changed=%d' % changed)
else:
    print('stale_mark_running noop')
PY2" || true
}

is_driver_busy() {
  if _is_match_running "run_wfa_queue_powered.py --queue"; then
    return 0
  fi
  if _is_match_running "watch_wfa_queue.sh --queue"; then
    return 0
  fi
  if _is_match_running "scripts/optimization/run_wfa_queue.py --queue"; then
    return 0
  fi

  # Check remote running markers; tolerate ssh failures as no extra workload.
  local remote_count
  remote_count="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=6 "$SERVER_USER@$SERVER_IP" "pgrep -f 'watch_wfa_queue.sh|run_wfa_queue.py|python.*walk_forward' | awk -v self=\"\$\$\" '\$1 != self' | wc -l" || true)"
  if [[ -n "$remote_count" && "$remote_count" -gt 0 ]]; then
    return 0
  fi

  return 1
}

start_queue() {
  local queue_rel="$1"
  local target="$ROOT_DIR/$queue_rel"

  if [[ ! -f "$target" ]]; then
    log "candidate_missing queue=$queue_rel"
    return 1
  fi

  local stamp
  stamp="$(date -u +%Y%m%d_%H%M%S)"
  local qlog="$STATE_DIR/run_${stamp}_$(basename "$(dirname "$queue_rel")").log"

  log "start queue_rel=$queue_rel"
  (
    cd "$ROOT_DIR"
    AUTONOMOUS_MODE=1 \
    ALLOW_HEAVY_RUN=1 \
    ./.venv/bin/python scripts/optimization/run_wfa_queue_powered.py \
      --queue "$queue_rel" \
      --compute-host "$SERVER_IP" \
      --ssh-user "$SERVER_USER" \
      --parallel 8 \
      --statuses auto \
      --max-retries 5 \
      --watchdog true \
      --wait-completion false \
      --postprocess true \
      --poweroff true \
      >>"$qlog" 2>&1
  ) &
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    log "failed_to_start queue=$queue_rel rc=$rc"
    return "$rc"
  fi
  echo "running queue=$queue_rel started_at=$stamp log=$qlog" > "$STATE_FILE"
  log "started queue=$queue_rel log=$qlog"
}

# Ensure queue rows are visible to local runner.
if ! find "$QUEUE_ROOT" -name run_queue.csv >/dev/null 2>&1; then
  echo "No aggregate queues found: $QUEUE_ROOT" | tee "$STATE_FILE"
  exit 1
fi

while true; do
  find_candidate
  IFS=',' read -r queue planned running stalled failed completed total priority < <(tail -n 1 "$CANDIDATE_FILE")

  if [[ -z "${queue:-}" || "$queue" == "queue" ]]; then
    echo "idle now=none completed=all" > "$STATE_FILE"
    log "no_pending_queues"
    sleep 180
    continue
  fi

  # Convert absolute path to repo-relative.
  queue_rel="${queue#$ROOT_DIR/}"

  if is_driver_busy; then
    echo "busy current_queue=$queue_rel running_or_stalled=$running,$stalled planned=$planned" > "$STATE_FILE"
    log "busy skip start queue_rel=$queue_rel running=$running stalled=$stalled planned=$planned"
    sleep 90
    continue
  fi

  if [[ "$running" -gt 0 ]]; then
    log "found_stale_running recover queue_rel=$queue_rel running=$running stalled=$stalled planned=$planned"
    if [[ "$planned" -eq 0 && "$stalled" -eq 0 ]]; then
      stale_running "$queue_rel"
    fi
  fi

  log "candidate queue=$queue_rel planned=$planned running=$running stalled=$stalled failed=$failed completed=$completed"
  start_queue "$queue_rel" || true

  # Backoff to let startup propagate before next decision.
  sleep 120

done
