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
ORPHAN_FILE="$STATE_DIR/orphan_queues.csv"
HEARTBEAT_STATE="$STATE_DIR/heartbeat_state.json"
STALE_RUNNING_SEC="${STALE_RUNNING_SEC:-900}"
ORPHAN_STALE_SECONDS="${ORPHAN_STALE_SECONDS:-600}"
ORPHAN_COOLDOWN_SECONDS="${ORPHAN_COOLDOWN_SECONDS:-1800}"
RUN_INDEX_PATH="${RUN_INDEX_PATH:-$ROOT_DIR/artifacts/wfa/aggregate/rollup/run_index.csv}"
PROMOTION_PRE_RANK_TOPK="${PROMOTION_PRE_RANK_TOPK:-8}"
PROMOTION_SELECTION_PROFILE="promote_profile"
PROMOTION_SELECTION_MODE="fullspan"
FULLSPAN_POLICY_NAME="fullspan_v1"
DECISION_NOTES_FILE="$STATE_DIR/decision_notes.jsonl"


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
  local skip_orphans="${1:-1}"
  local orphan_path=""
  if [[ "$skip_orphans" == "1" && -f "$ORPHAN_FILE" ]]; then
    orphan_path="$ORPHAN_FILE"
  fi

  python3 - "$QUEUE_ROOT" "$CANDIDATE_FILE" "$orphan_path" "$RUN_INDEX_PATH" "$PROMOTION_PRE_RANK_TOPK" <<'PY'
import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


def to_float(value, default=None):
    try:
        return float((value or '').strip())
    except Exception:
        return default


def is_true(value):
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def hard_gate_pass(row):
    if not row:
        return False, "missing_row"
    if not is_true(row.get("metrics_present", "")):
        return False, "metrics_missing"

    total_trades = to_float(row.get("total_trades"), 0.0)
    total_pairs = to_float(row.get("total_pairs_traded"), 0.0)
    worst_dd = abs(to_float(row.get("max_drawdown_on_equity"), 0.0) or 0.0)
    worst_robust_pnl = to_float(row.get("total_pnl"), 0.0)
    worst_step = to_float(row.get("tail_loss_worst_period_pnl", 0), to_float(row.get("tail_loss_worst_pair_pnl", 0), 0.0))

    if total_trades < 200:
        return False, "trades_gate_fail"
    if total_pairs < 20:
        return False, "pairs_gate_fail"
    if worst_dd > 0.50:
        return False, "dd_gate_fail"
    if worst_robust_pnl < 0:
        return False, "economic_gate_fail"
    if worst_step < -200:
        return False, "step_gate_fail"
    return True, "pass"


def canonical_base(run_id):
    return re.sub(r'^(holdout_|stress_)', '', run_id)

queue_root = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
orphan_file = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
run_index_path = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
pre_rank_top_k = int(sys.argv[5]) if len(sys.argv) > 5 and str(sys.argv[5]).strip() else 8

now = time.time()

run_index = {}
if run_index_path and run_index_path.exists():
    try:
        with run_index_path.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                run_index[row.get('run_id', '').strip()] = row
    except Exception:
        run_index = {}

orphan = {}
if orphan_file and orphan_file.exists():
    try:
        with orphan_file.open(newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                q = (row.get('queue') or '').strip()
                try:
                    until = float((row.get('until_ts') or '').strip() or 0)
                except Exception:
                    until = 0
                if q:
                    orphan[q] = until
    except Exception:
        orphan = {}

out = []
for p in sorted(queue_root.rglob('run_queue.csv')):
    try:
        rows = list(csv.DictReader(p.open(newline='')))
    except Exception:
        continue

    st = defaultdict(int)
    for r in rows:
        st[(r.get('status') or '').strip().lower()] += 1

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

    queue_rel = str(p)
    if queue_rel in orphan:
        try:
            if now < float(orphan.get(queue_rel, 0.0)):
                continue
        except Exception:
            pass

    mtime = float(p.stat().st_mtime)
    age_min = max(0.0, (now - mtime) / 60.0)

    by_base = defaultdict(lambda: {'holdout': None, 'stress': None, 'status': None})

    for row in rows:
        results_dir = (row.get('results_dir') or '').strip()
        if not results_dir:
            continue
        run_id = Path(results_dir).name.strip()
        base = canonical_base(run_id)
        if run_id.startswith('holdout_'):
            by_base[base]['holdout'] = run_index.get(run_id)
        elif run_id.startswith('stress_'):
            by_base[base]['stress'] = run_index.get(run_id)
        else:
            by_base[base]['holdout'] = run_index.get(run_id)
        by_base[base]['status'] = (row.get('status') or '').strip().lower()

    pass_configs = 0
    fail_configs = 0
    unknown_configs = 0
    fail_reasons = []

    for base, bundle in by_base.items():
        h = bundle.get('holdout')
        s = bundle.get('stress')

        if h is None and s is None:
            unknown_configs += 1
            continue

        h_ok = True
        s_ok = True

        if h is None:
            h_ok = False
        else:
            ok, reason = hard_gate_pass(h)
            if not ok:
                fail_reasons.append(reason)
                h_ok = False

        if s is None:
            s_ok = False
        else:
            ok, reason = hard_gate_pass(s)
            if not ok:
                fail_reasons.append(reason)
                s_ok = False

        if h_ok and s_ok:
            pass_configs += 1
        elif (not h_ok) and (not s_ok):
            fail_configs += 1
        else:
            unknown_configs += 1

    total_configs = pass_configs + fail_configs + unknown_configs

    if total_configs == 0:
        promotion_potential = "UNKNOWN"
        gate_status = "OPEN"
        gate_reason = "insufficient_history"
    elif pass_configs == 0 and fail_configs > 0:
        promotion_potential = "REJECT"
        gate_status = "HARD_FAIL"
        gate_reason = fail_reasons[0] if fail_reasons else "hard_gate_fail"
    else:
        promotion_potential = "POSSIBLE"
        gate_status = "OPEN"
        gate_reason = "history_probe"

    stale_count = failed
    pre_rank_score = (
        pass_configs * 14.0
        + unknown_configs * 2.0
        - fail_configs * 4.5
        - stale_count * 1.2
        - age_min * 0.3
    )

    urgency = (stalled * 100.0) + (running * 20.0) + (age_min * 0.1) + (1.0 if pending > 0 else 0.0)

    if gate_status == "HARD_FAIL" and promotion_potential == "REJECT":
        pre_rank_score -= 10000

    out.append((
        -pre_rank_score,
        -urgency,
        -int(mtime),
        queue_rel,
        planned,
        running,
        stalled,
        failed,
        completed,
        total,
        f"{urgency:.3f}",
        promotion_potential,
        gate_status,
        gate_reason,
        f"{pre_rank_score:.3f}",
    ))

out.sort()

with out_csv.open('w', encoding='utf-8', newline='') as f:
    f.write('queue,planned,running,stalled,failed,completed,total,urgency,mtime,promotion_potential,gate_status,gate_reason,pre_rank_score\n')
    if not out:
        raise SystemExit(0)

    out = out[:pre_rank_top_k]
    _, __, ___, queue, planned, running, stalled, failed, completed, total, urgency, potential, gate_status, gate_reason, pre_rank = out[0]
    f.write(f"{queue},{planned},{running},{stalled},{failed},{completed},{total},{urgency},{int(Path(queue).stat().st_mtime)},{potential},{gate_status},{gate_reason},{pre_rank}\n")
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

  if [[ -n "$pattern" ]]; then
    echo "$pattern"
    return
  fi

  echo "UNKNOWN"
}

mark_orphan() {
  local queue_rel="$1"
  local reason="$2"
  local until_ts
  until_ts="$(( $(date +%s) + ORPHAN_COOLDOWN_SECONDS ))"

  python3 - "$ORPHAN_FILE" "$queue_rel" "$until_ts" "$reason" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
queue = sys.argv[2]
until_ts = float(sys.argv[3])
reason = sys.argv[4]

rows = []
if path.exists():
    try:
        with path.open(newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if (r.get('queue') or '').strip() == queue:
                    continue
                rows.append(r)
    except Exception:
        rows = []

rows.append({'queue': queue, 'until_ts': f'{until_ts:.0f}', 'reason': reason})
path.parent.mkdir(parents=True, exist_ok=True)
with path.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['queue', 'until_ts', 'reason'])
    w.writeheader()
    for r in rows:
        w.writerow(r)
PY
  log "orphan_marked queue=$queue_rel until_ts=$until_ts reason=$reason"
}

clear_orphan() {
  local queue_rel="$1"
  python3 - "$ORPHAN_FILE" "$queue_rel" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
queue = sys.argv[2]

if not path.exists():
    raise SystemExit(0)

rows = []
with path.open(newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if (r.get('queue') or '').strip() != queue:
            rows.append(r)

if rows:
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['queue', 'until_ts', 'reason'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
else:
    try:
        path.unlink()
    except Exception:
        pass
PY
}

cleanup_orphans() {
  python3 - "$ORPHAN_FILE" <<'PY'
import csv
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
now = time.time()
if not path.exists():
    raise SystemExit(0)

rows = []
with path.open(newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        try:
            if now < float((r.get('until_ts') or '0').strip()):
                rows.append(r)
        except Exception:
            continue

if rows:
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['queue', 'until_ts', 'reason'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
else:
    try:
        path.unlink()
    except Exception:
        pass
PY
}

log_decision_note() {
  local queue="$1"
  local action="$2"
  local reason="$3"
  local next_step="$4"

  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  python3 - "$DECISION_NOTES_FILE" "$queue" "$action" "$reason" "$next_step" "$ts" <<'PY'
import json
import sys
from pathlib import Path

path, queue, action, reason, next_step, ts = sys.argv[1:7]
p = Path(path)
p.parent.mkdir(parents=True, exist_ok=True)
rec = {
    "ts": ts,
    "queue": queue,
    "action": action,
    "reason": reason,
    "next_step": next_step,
}
with p.open("a", encoding="utf-8") as f:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
PY
}

heartbeat_update() {
  local queue_rel="$1"
  local pending="$2"
  local completed="$3"
  local total="$4"
  local planned="$5"
  local running="$6"
  local stalled="$7"
  local out

  # Shell-visible heartbeat state for downstream policy decisions.
  hb_queue=""
  hb_stale_sec="0"

  out="$(python3 "$ROOT_DIR/scripts/optimization/_autonomous_heartbeat.py"     --state "$HEARTBEAT_STATE"     --queue "$queue_rel"     --pending "$pending"     --completed "$completed"     --total "$total"     --planned "$planned"     --running "$running"     --stalled "$stalled" 2>/dev/null || true)"

  if [[ -z "$out" ]]; then
    return
  fi

  IFS='|' read -r hb_queue hb_pending hb_completed hb_rate hb_eta hb_stale hb_done <<< "$out"
  hb_stale_sec="$hb_stale"
  log "heartbeat queue=$hb_queue pending=$hb_pending completed=$hb_completed rate_per_min=$hb_rate eta_min=$hb_eta stale_sec=$hb_stale done=$hb_done"

  if [[ "$hb_done" == "1" ]]; then
    clear_orphan "$queue_rel"
  fi
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

repair_stalled_queue() {
  local queue_rel="$1"
  local planned="$2"
  local running="$3"
  local stalled="$4"
  local total="$5"
  local cause="$6"

  local parallel
  parallel="$(choose_parallel "$planned" "$running" "$stalled" "$total" "$cause")"
  if (( parallel < 2 )); then
    parallel=2
  fi
  if (( parallel > 12 )); then
    parallel=12
  fi

  log "repair_stalled queue=$queue_rel parallel=$parallel stale_sec=$hb_stale_sec cause=$cause"
  local rc=0
  ("$ROOT_DIR/scripts/optimization/recover_stalled_queue.sh" --queue "$queue_rel" --parallel "$parallel") >>"$LOG_FILE" 2>&1 || rc=$?

  if [[ "$rc" -eq 0 ]]; then
    log "repair_stalled_success queue=$queue_rel"
    clear_orphan "$queue_rel"
    return 0
  fi

  log "repair_stalled_failed queue=$queue_rel rc=$rc"
  return "$rc"
}

repair_stalled_state_record() {
  local queue_rel="$1"
  local cur="${stalled_repair_failures[$queue_rel]:-0}"
  local new_failures=$((cur + 1))
  stalled_repair_failures["$queue_rel"]="$new_failures"
  echo "$new_failures"
}

repair_stalled_state_reset() {
  local queue_rel="$1"
  stalled_repair_failures["$queue_rel"]=0
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
  log_state "running queue=$queue_rel reason=$cause started_at=$stamp log=$qlog parallel=$parallel max_retries=$max_retries selection_policy=$FULLSPAN_POLICY_NAME selection_mode=$PROMOTION_SELECTION_MODE promotion_verdict=$promotion_verdict pre_rank_score=$pre_rank_score promotion_potential=$promotion_potential gate_status=$gate_status"
  log "started queue=$queue_rel log=$qlog selection_policy=$FULLSPAN_POLICY_NAME selection_mode=$PROMOTION_SELECTION_MODE promotion_verdict=$promotion_verdict"
}

# lightweight per-queue progress cache for stale detection in-shell
declare -A last_pending_by_queue

cleanup_orphans

declare -A stalled_repair_failures

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
  find_candidate 1

  if [[ ! -s "$CANDIDATE_FILE" ]]; then
    cleanup_orphans
    find_candidate 0
  fi

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

  IFS=',' read -r queue planned running stalled failed completed total urgency mtime promotion_potential gate_status gate_reason pre_rank_score < <(tail -n 1 "$CANDIDATE_FILE")

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

  promotion_verdict="ANALYZE"
  if [[ "$promotion_potential" == "REJECT" && "$gate_status" == "HARD_FAIL" ]]; then
    promotion_verdict="REJECT"
  fi

  if [[ -n "$prev_queue" && "$prev_queue" == "$queue_rel" ]]; then
    if [[ "$pending" -lt "$prev_pending" ]]; then
      clear_orphan "$queue_rel"
      log "progress_seen queue=$queue_rel prev=$prev_pending curr=$pending"
      sync_queue_status "$queue_rel"
    fi
  fi

  heartbeat_update "$queue_rel" "$pending" "$completed" "$total" "$planned" "$running" "$stalled"

  if [[ "$promotion_verdict" == "REJECT" && "$promotion_potential" == "REJECT" ]]; then
    log "candidate_gated_reject queue=$queue_rel promotion_verdict=$promotion_verdict gate_status=$gate_status gate_reason=$gate_reason pre_rank_score=$pre_rank_score"
    log_decision_note "$queue_rel" "REJECT" "gate_status=$gate_status reason=$gate_reason" "skip_and_select_next_candidate"
    sleep "$adaptive_idle_sleep"
    continue
  fi

  prev_queue="$queue_rel"
  prev_planned="$planned"
  prev_running="$running"
  prev_stalled="$stalled"
  prev_failed="$failed"
  prev_pending="$pending"
  action=""
  reason=""

  if [[ -n "$hb_queue" && "$running" -eq 0 && "$stalled" -gt 0 ]]; then
    if awk -v stale="$hb_stale_sec" -v th="$ORPHAN_STALE_SECONDS" 'BEGIN { exit (stale + 0 >= th) ? 0 : 1 }'; then
      reason="stalled_queue_no_progress"
      if repair_stalled_queue "$queue_rel" "$planned" "$running" "$stalled" "$total" "UNKNOWN"; then
        action="REPAIRED_STALLED"
        repair_stalled_state_reset "$queue_rel"
        log_decision_note "$queue_rel" "REPAIRED_STALLED" "repair_stalled_success" "reselect_next_candidate"
      else
        fails=$(repair_stalled_state_record "$queue_rel")
        if (( fails >= 3 )); then
          mark_orphan "$queue_rel" "stalled_repair_fail_${fails}"
          action="FAIL_CLOSED"
          repair_stalled_state_reset "$queue_rel"
          log_decision_note "$queue_rel" "FAIL_CLOSED" "repair_stalled_failures_exhausted" "orphan_and_select_next"
        else
          action="REPAIR_STALLED_FAIL"
        fi
      fi
      log "stalled_repair_cycle queue=$queue_rel action=$action stale_sec=$hb_stale_sec"
    fi
  fi

  if [[ -z "$reason" ]]; then
    if [[ "$running" -gt 0 && "$planned" -eq 0 && "$stalled" -eq 0 ]]; then
      reason="stale_running_repair"
      stale_out=$(stale_running "$queue_rel")
      action="RUN_REPAIR"
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
  fi

  if [[ -z "$action" ]]; then
    action="ANALYZE"
  fi

  cause=""
  if [[ "$reason" == "no_pending" ]]; then
    cause="IDLE"
  elif [[ "$reason" == "stale_running_repair" ]]; then
    cause="REPAIR"
  elif [[ "$reason" == "stalled_queue_no_progress" ]]; then
    cause="REPAIR"
  else
    cause="$(classify_root_cause "$queue_rel" "$reason" "$queue_rel")"
  fi

  if [[ "$action" == "REPAIR_STALLED_FAIL" || "$action" == "REPAIRED_STALLED" || "$action" == "FAIL_CLOSED" ]]; then
    log "repair_gate queue=$queue_rel action=$action"
    sleep 10
    continue
  fi

  if is_driver_busy; then
    if [[ -n "$prev_queue" && "$prev_queue" == "$queue_rel" ]]; then
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
    log "busy skip start queue_rel=$queue_rel urgency=$urgency reason=$reason pending=$pending repeat=$busy_repeat_count cause=$cause action=$action"
    continue
  fi

  busy_repeat_count=0
  adaptive_idle_sleep=30

  if [[ "$reason" == "no_pending" ]]; then
    log_state "idle current_queue=$queue_rel pending=0 completed=$completed"
    log "no_pending queue=$queue_rel action=WAIT selection_policy=$FULLSPAN_POLICY_NAME selection_mode=$PROMOTION_SELECTION_MODE promotion_verdict=$promotion_verdict"
    sleep "$adaptive_idle_sleep"
    continue
  fi

  if [[ -z "$cause" ]]; then
    cause="UNKNOWN"
  fi

  log "candidate queue=$queue_rel reason=$reason cause=$cause urgency=$urgency planned=$planned running=$running stalled=$stalled failed=$failed completed=$completed action=$action selection_policy=$FULLSPAN_POLICY_NAME selection_mode=$PROMOTION_SELECTION_MODE selection_profile=$PROMOTION_SELECTION_PROFILE promotion_verdict=$promotion_verdict gate_status=$gate_status gate_reason=$gate_reason promotion_potential=$promotion_potential pre_rank_score=$pre_rank_score"
  if ! start_queue "$queue_rel" "$cause" "$planned" "$running" "$stalled" "$total"; then
    log "start_failed queue=$queue_rel cause=$cause"
  fi

  sleep 120

done
