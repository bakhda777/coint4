#!/usr/bin/env bash
set -euo pipefail

# Lightweight orchestrator: keeps advancing through non-terminal run_queues.
# It picks the next queue with the highest work urgency and starts
# run_wfa_queue_powered without manual intervention.

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
QUEUE_ROOT="$ROOT_DIR/artifacts/wfa/aggregate"
STATE_DIR="$ROOT_DIR/artifacts/wfa/aggregate/.autonomous"
STATE_FILE="$STATE_DIR/driver_state.txt"
LOG_FILE="$STATE_DIR/driver.log"
SERVER_IP="${SERVER_IP:-85.198.90.128}"
SERVER_USER="${SERVER_USER:-root}"
LOCK_FILE="$STATE_DIR/driver.lock"
CANDIDATE_FILE="$STATE_DIR/candidate.csv"
STALE_RUNNING_SEC="${STALE_RUNNING_SEC:-900}"

mkdir -p "$STATE_DIR"

log() {
  printf '%s | %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*" >> "$LOG_FILE"
}

log_state() {
  echo "$*" > "$STATE_FILE"
}

# Single driver instance guard.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "driver_already_running"
  exit 0
fi
trap 'flock -u 9; rm -f "$LOCK_FILE"' EXIT

find_candidate() {
  python3 - "$QUEUE_ROOT" "$CANDIDATE_FILE" <<'PY'
import csv
import sys
import time
from collections import Counter
from pathlib import Path

queue_root = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
now = time.time()
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
    completed = int(st.get('completed', 0))
    total = len(rows)

    if total == 0:
        continue

    pending = planned + running + stalled + failed
    if pending == 0:
        continue

    mtime = float(p.stat().st_mtime)
    age_min = max(0.0, (now - mtime) / 60.0)
    urgency = (stalled * 100.0) + (running * 20.0) + (age_min * 0.1)

    out.append((
        -urgency,
        -int(p.stat().st_mtime),
        str(p),
        planned,
        running,
        stalled,
        failed,
        completed,
        total,
        f"{urgency:.3f}",
    ))

out.sort()

with out_csv.open('w', encoding='utf-8', newline='') as f:
    f.write('queue,planned,running,stalled,failed,completed,total,urgency,mtime\n')
    if not out:
        sys.exit(0)
    _, __, queue, planned, running, stalled, failed, completed, total, urgency = out[0]
    f.write(f"{queue},{planned},{running},{stalled},{failed},{completed},{total},{urgency},{int(Path(queue).stat().st_mtime)}\n")
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
  local changed=0

  local stale_sec="${STALE_RUNNING_SEC}"
  [[ -z "$stale_sec" ]] && stale_sec=900
  if [[ "$stale_sec" -lt 60 ]]; then
    stale_sec=60
  fi

  local local_changed=0
  local_changed=$(python3 "$ROOT_DIR/scripts/optimization/_autonomous_stale_running.py" --queue "$ROOT_DIR/$queue_rel" --stale-sec "$stale_sec")
  changed=$((changed + local_changed))

  local remote_changed=0
  remote_changed=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8 "$SERVER_USER@$SERVER_IP" "python3 /opt/coint4/coint4/scripts/optimization/_autonomous_stale_running.py --queue /opt/coint4/coint4/$queue_rel --stale-sec '$stale_sec' --root /opt/coint4/coint4" || true)
  changed=$((changed + remote_changed))

  if [[ "$changed" -gt 0 ]]; then
    log "stale_running_marked queue=$queue_rel changed=$changed stale_sec=$stale_sec"
    echo "recovered_running_to_stalled queue=$queue_rel changed=$changed"
    return 0
  fi

  echo "stale_running_nochange queue=$queue_rel"
  return 0
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

  local remote_count
  remote_count="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=6 "$SERVER_USER@$SERVER_IP" "pgrep -f 'watch_wfa_queue.sh|run_wfa_queue.py|python.*walk_forward' | awk -v self=\"\$\$\" '\$1 != self' | wc -l" || true)"
  if [[ -n "$remote_count" && "$remote_count" -gt 0 ]]; then
    return 0
  fi

  return 1
}

sync_queue_status() {
  local queue_rel="$1"
  local target="$ROOT_DIR/$queue_rel"
  if [[ ! -f "$target" ]]; then
    return 0
  fi

  (cd "$ROOT_DIR" && PYTHONPATH=src ./.venv/bin/python scripts/optimization/sync_queue_status.py --queue "$queue_rel") >>"$LOG_FILE" 2>&1 || true
  log "sync_queue_status queue=$queue_rel done"
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
  log_state "running queue=$queue_rel started_at=$stamp log=$qlog"
  log "started queue=$queue_rel log=$qlog"
}

adaptive_idle_sleep=30
prev_queue=""
prev_planned=0
prev_running=0
prev_stalled=0
prev_failed=0

if ! find "$QUEUE_ROOT" -name run_queue.csv >/dev/null 2>&1; then
  log_state "No aggregate queues found: $QUEUE_ROOT"
  exit 1
fi

while true; do
  find_candidate

  if [[ ! -s "$CANDIDATE_FILE" ]]; then
    log_state "idle now=none completed=all"
    log "candidate_empty"
    if [[ "$adaptive_idle_sleep" -lt 300 ]]; then
      adaptive_idle_sleep=$((adaptive_idle_sleep * 2))
    fi
    if [[ "$adaptive_idle_sleep" -gt 300 ]]; then
      adaptive_idle_sleep=300
    fi
    sleep "$adaptive_idle_sleep"
    continue
  fi

  IFS=',' read -r queue planned running stalled failed completed total urgency mtime < <(tail -n 1 "$CANDIDATE_FILE")

  if [[ -z "${queue:-}" || "$queue" == "queue" ]]; then
    log_state "idle now=none completed=all"
    log "candidate_parse_empty"
    if [[ "$adaptive_idle_sleep" -lt 300 ]]; then
      adaptive_idle_sleep=$((adaptive_idle_sleep * 2))
    fi
    if [[ "$adaptive_idle_sleep" -gt 300 ]]; then
      adaptive_idle_sleep=300
    fi
    sleep "$adaptive_idle_sleep"
    continue
  fi

  queue_rel="${queue#$ROOT_DIR/}"

  if [[ -n "$prev_queue" && "$prev_queue" == "$queue_rel" ]]; then
    prev_pending=$((prev_planned + prev_running + prev_stalled + prev_failed))
    curr_pending=$((planned + running + stalled + failed))
    if [[ "$curr_pending" -lt "$prev_pending" ]]; then
      log "progress_seen queue=$queue_rel prev=$prev_pending curr=$curr_pending"
      sync_queue_status "$queue_rel"
    fi
  fi

  prev_queue="$queue_rel"
  prev_planned="$planned"
  prev_running="$running"
  prev_stalled="$stalled"
  prev_failed="$failed"

  if is_driver_busy; then
    adaptive_idle_sleep=30
    log_state "busy current_queue=$queue_rel running_or_stalled=$running,$stalled planned=$planned"
    log "busy skip start queue_rel=$queue_rel urgency=$urgency running=$running stalled=$stalled planned=$planned"
    sleep 90
    continue
  fi

  adaptive_idle_sleep=30

  reason=""
  if [[ "$running" -gt 0 && "$planned" -eq 0 && "$stalled" -eq 0 ]]; then
    reason="stale_running_repair"
    stale_out=$(stale_running "$queue_rel")
    log "run_repair queue=$queue_rel out=\"$stale_out\""
  elif [[ "$running" -gt 0 ]]; then
    reason="active_running"
  elif [[ "$stalled" -gt 0 ]]; then
    reason="stalled_queue"
  elif [[ "$planned" -gt 0 ]]; then
    reason="planned_work"
  else
    reason="no_pending"
  fi

  if [[ "$reason" == "no_pending" ]]; then
    log_state "idle current_queue=$queue_rel pending=0 completed=$completed"
    log "no_pending queue=$queue_rel"
    sleep "$adaptive_idle_sleep"
    continue
  fi

  log "candidate queue=$queue_rel reason=$reason urgency=$urgency planned=$planned running=$running stalled=$stalled failed=$failed completed=$completed"
  start_queue "$queue_rel" || true

  sleep 120

done
