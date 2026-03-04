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

get_vps_load() {
  local load1=""
  load1="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=6 "$SERVER_USER@$SERVER_IP" "cat /proc/loadavg 2>/dev/null | awk '{print \\$1}'" || true)"
  if [[ -z "$load1" ]]; then
    echo 0
    return
  fi
  echo "$load1"
}

choose_parallel() {
  local planned="$1"
  local running="$2"
  local stalled="$3"
  local total="$4"
  local cause="$5"

  local pending=$((planned + running + stalled))
  local load1
  local p

  load1="$(get_vps_load)"

  # Heavy and noisy queues run more conservatively.
  if (( total >= 100 || pending >= 40 )); then
    p=4
  elif (( total <= 24 )); then
    if awk -v v="$load1" 'BEGIN { exit (v >= 12.0) ? 0 : 1 }'; then
      p=8
    else
      p=12
    fi
  elif (( total <= 60 )); then
    p=8
  else
    p=6
  fi

  # cause-aware tuning
  case "$cause" in
    NETWORK)
      p=4
      ;;
    MODEL|TIMEOUT)
      p=4
      ;;
    DATA)
      p=6
      ;;
  esac

  # never exceed 12 in autonomous mode
  if (( p > 12 )); then
    p=12
  fi
  if (( p < 2 )); then
    p=2
  fi

  echo "$p"
}

choose_max_retries() {
  local cause="$1"
  case "$cause" in
    NETWORK)
      echo 2
      ;;
    DATA)
      echo 1
      ;;
    TIMEOUT)
      echo 3
      ;;
    MODEL)
      echo 2
      ;;
    *)
      echo 5
      ;;
  esac
}

classify_root_cause() {
  local queue_rel="$1"
  local pattern="$2"
  local qdir="$3"

  local latest_log=""
  local latest=""
  local group

  group="$(basename "$qdir")"
  latest_log="$(ls -t "$STATE_DIR"/run_*.log 2>/dev/null | grep -F "$group" | head -n 1 || true)"
  if [[ -z "$latest_log" ]]; then
    echo "UNKNOWN"
    return
  fi

  latest="$(tail -n 120 "$latest_log" 2>/dev/null || true)"

  # Also inspect queue-local error files if present.
  local err_file=""
  if [[ -n "$qdir" ]]; then
    if [[ -f "$ROOT_DIR/$qdir/strategy_metrics/equity_curve/errors" ]]; then
      err_file="$ROOT_DIR/$qdir/strategy_metrics/equity_curve/errors"
    elif [[ -f "$ROOT_DIR/$qdir/strategy_metrics/equity_curve_error.log" ]]; then
      err_file="$ROOT_DIR/$qdir/strategy_metrics/equity_curve_error.log"
    fi
    if [[ -n "$err_file" ]]; then
      latest="$latest\n$(tail -n 40 "$err_file" 2>/dev/null || true)"
    fi
  fi

  if printf '%s' "$latest" | grep -Eqi '(network|connection.*timeout|connection reset|temporarily unavailable|failed to connect|ssh:|timed out)'; then
    echo "NETWORK"
    return
  fi

  if printf '%s' "$latest" | grep -Eqi '(file not found|filenotfound|yaml|missing.*config|keyerror|config.*error|permission denied|No such file|ошибка.*данн)'; then
    echo "DATA"
    return
  fi

  if printf '%s' "$latest" | grep -Eqi '(equity_series is empty|RuntimeError|Traceback|ValueError|TypeError|IndexError|No module|model.*error|empty.*equity|nan|inf)'; then
    echo "MODEL"
    return
  fi

  if printf '%s' "$latest" | grep -Eqi '(timeout|timed out|deadline|time limit|watchdog)'; then
    echo "TIMEOUT"
    return
  fi

  # pattern-based reason from caller
  if [[ -n "$pattern" ]]; then
    echo "$pattern"
    return
  fi

  echo "UNKNOWN"
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
  remote_changed=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8 "$SERVER_USER@$SERVER_IP" "python3 - <<'PY'
import csv
import pathlib
import time

queue_path = pathlib.Path('/opt/coint4/coint4') / '$queue_rel'
stale_sec = float('$stale_sec')
if not queue_path.exists():
    print(0)
    raise SystemExit(0)
rows = list(csv.DictReader(queue_path.open(newline='')))
now = time.time()
changed = 0
for row in rows:
    status = (row.get('status') or '').strip().lower()
    if status != 'running':
        continue

    results_dir = str(row.get('results_dir') or '').strip()
    run_dir = pathlib.Path(results_dir) if results_dir else None
    if run_dir and not run_dir.is_absolute():
        run_dir = pathlib.Path('/opt/coint4/coint4') / run_dir

    mtimes = []
    if run_dir and run_dir.exists():
        for name in ('strategy_metrics.csv', 'equity_curve.csv', 'canonical_metrics.json', 'progress.json', 'status.json'):
            c = run_dir / name
            if c.exists():
                try:
                    mtimes.append(c.stat().st_mtime)
                except Exception:
                    pass
        for patt in ('*.log', '*.json', '*.csv'):
            for c in run_dir.glob(patt):
                if not c.is_file():
                    continue
                try:
                    mtimes.append(c.stat().st_mtime)
                except Exception:
                    pass

    if not mtimes:
        row['status'] = 'stalled'
        changed += 1
        continue

    age_sec = int(max(0.0, now - max(mtimes)))
    if age_sec >= stale_sec:
        row['status'] = 'stalled'
        changed += 1

if changed > 0 and rows:
    with queue_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

print(changed)
PY" || true)
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
  local cause="$2"
  local target="$ROOT_DIR/$queue_rel"

  if [[ ! -f "$target" ]]; then
    log "candidate_missing queue=$queue_rel"
    return 1
  fi

  local planned running stalled total
  planned="${3:-0}"
  running="${4:-0}"
  stalled="${5:-0}"
  total="${6:-0}"

  local parallel
  local max_retries

  parallel="$(choose_parallel "$planned" "$running" "$stalled" "$total" "$cause")"
  max_retries="$(choose_max_retries "$cause")"

  local stamp
  stamp="$(date -u +%Y%m%d_%H%M%S)"
  local qlog="$STATE_DIR/run_${stamp}_$(basename "$(dirname "$queue_rel")").log"

  log "start queue_rel=$queue_rel cause=$cause parallel=$parallel max_retries=$max_retries"
  (
    cd "$ROOT_DIR"
    AUTONOMOUS_MODE=1 \
    ALLOW_HEAVY_RUN=1 \
    ./.venv/bin/python scripts/optimization/run_wfa_queue_powered.py \
      --queue "$queue_rel" \
      --compute-host "$SERVER_IP" \
      --ssh-user "$SERVER_USER" \
      --parallel "$parallel" \
      --statuses auto \
      --max-retries "$max_retries" \
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
  log_state "running queue=$queue_rel reason=$cause started_at=$stamp log=$qlog parallel=$parallel max_retries=$max_retries"
  log "started queue=$queue_rel log=$qlog"
}

adaptive_idle_sleep=30
prev_queue=""
prev_planned=0
prev_running=0
prev_stalled=0
prev_failed=0
prev_pending=0
busy_repeat_count=0
busy_backoff_seconds=90

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
  pending=$((planned + running + stalled + failed))

  if [[ -n "$prev_queue" && "$prev_queue" == "$queue_rel" ]]; then
    if [[ "$pending" -lt "$prev_pending" ]]; then
      log "progress_seen queue=$queue_rel prev=$prev_pending curr=$pending"
      sync_queue_status "$queue_rel"
    fi
  fi

  prev_queue="$queue_rel"
  prev_planned="$planned"
  prev_running="$running"
  prev_stalled="$stalled"
  prev_failed="$failed"
  prev_pending="$pending"

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

  cause=""
  if [[ "$reason" == "no_pending" ]]; then
    cause="IDLE"
  elif [[ "$reason" == "stale_running_repair" ]]; then
    cause="REPAIR"
  else
    cause="$(classify_root_cause "$queue_rel" "$reason" "$queue_rel")"
  fi

  if is_driver_busy; then
    if [[ "$prev_queue" == "$queue_rel" ]]; then
      if (( pending == prev_pending )); then
        busy_repeat_count=$((busy_repeat_count + 1))
      else
        busy_repeat_count=1
      fi
    else
      busy_repeat_count=1
    fi

    if (( busy_repeat_count >= 3 )); then
      log "busy_throttle queue=$queue_rel repeat=$busy_repeat_count pending=$pending cause=$cause"
      sync_queue_status "$queue_rel" || true
      sleep 90
    else
      sleep "$busy_backoff_seconds"
    fi
    log_state "busy current_queue=$queue_rel running_or_stalled=$running,$stalled planned=$planned reason=$reason"
    log "busy skip start queue_rel=$queue_rel urgency=$urgency reason=$reason pending=$pending repeat=$busy_repeat_count cause=$cause"
    continue
  fi

  busy_repeat_count=0
  adaptive_idle_sleep=30

  if [[ "$reason" == "no_pending" ]]; then
    log_state "idle current_queue=$queue_rel pending=0 completed=$completed"
    log "no_pending queue=$queue_rel"
    sleep "$adaptive_idle_sleep"
    continue
  fi

  if [[ -z "$cause" ]]; then
    cause="UNKNOWN"
  fi

  log "candidate queue=$queue_rel reason=$reason cause=$cause urgency=$urgency planned=$planned running=$running stalled=$stalled failed=$failed completed=$completed"
  if ! start_queue "$queue_rel" "$cause" "$planned" "$running" "$stalled" "$total"; then
    log "start_failed queue=$queue_rel cause=$cause"
  fi

  # Let startup propagate before next decision.
  sleep 120

done
