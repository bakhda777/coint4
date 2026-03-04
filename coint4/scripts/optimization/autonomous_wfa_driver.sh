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


FULLSPAN_CYCLE_STATE_FILE="$STATE_DIR/mini_cycle_state.txt"
FULLSPAN_CYCLE_CACHE_FILE="$STATE_DIR/fullspan_cycle_cache.json"
FULLSPAN_DECISION_STATE_FILE="$STATE_DIR/fullspan_decision_state.json"
FULLSPAN_CONFIRM_MIN_GROUPS="${FULLSPAN_CONFIRM_MIN_GROUPS:-2}"
FULLSPAN_CONFIRM_MIN_REPLIES="${FULLSPAN_CONFIRM_MIN_REPLIES:-2}"
NO_PROGRESS_STALE_CYCLES="${NO_PROGRESS_STALE_CYCLES:-6}"

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

  python3 - "$QUEUE_ROOT" "$CANDIDATE_FILE" "$orphan_path" "$RUN_INDEX_PATH" "$PROMOTION_PRE_RANK_TOPK" "$FULLSPAN_DECISION_STATE_FILE" <<'PY'
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


INITIAL_CAPITAL = 1000.0
HARDFAIL_MAP = {
    "metrics_missing": "METRICS_MISSING",
    "trades_gate_fail": "TRADES_FAIL",
    "pairs_gate_fail": "PAIRS_FAIL",
    "dd_gate_fail": "DD_FAIL",
    "economic_gate_fail": "ECONOMIC_FAIL",
    "step_gate_fail": "STEP_FAIL",
}


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
        return False, "metrics_missing"
    if not is_true(row.get("metrics_present", "")):
        return False, "metrics_missing"

    total_trades = to_float(row.get("total_trades"), 0.0)
    total_pairs = to_float(row.get("total_pairs_traded"), 0.0)
    worst_dd = abs(to_float(row.get("max_drawdown_on_equity"), 0.0) or 0.0)
    worst_robust_pnl = to_float(row.get("total_pnl"), 0.0)
    worst_step = to_float(
        row.get("tail_loss_worst_period_pnl"),
        to_float(row.get("tail_loss_worst_pair_pnl", 0), 0.0),
    )

    if total_trades < 200:
        return False, "trades_gate_fail"
    if total_pairs < 20:
        return False, "pairs_gate_fail"
    if worst_dd > 0.50:
        return False, "dd_gate_fail"
    if worst_robust_pnl < 0:
        return False, "economic_gate_fail"
    if worst_step < (-0.20 * INITIAL_CAPITAL):
        return False, "step_gate_fail"
    return True, "pass"


def quantile(values, q):
    vals = sorted(v for v in values if v is not None and v == v)
    if not vals:
        return None
    if not vals:
        return None
    n = len(vals)
    if n == 1:
        return vals[0]
    idx = min(max(int((n - 1) * q), 0), n - 1)
    return vals[idx]


def score_fullspan_v1_like(sharpes, tails):
    if not sharpes or not tails:
        return None
    try:
        worst_robust = min(sharpes)
        q20 = quantile(tails, 0.20)
        worst_step = min(tails)
        if q20 is None:
            q20 = worst_step
        return (
            float(worst_robust)
            - (2.0 * max(0.0, (-q20 / INITIAL_CAPITAL) - 0.03))
            - (1.0 * max(0.0, (-worst_step / INITIAL_CAPITAL) - 0.10))
        )
    except Exception:
        return None


def canonical_base(run_id):
    return re.sub(r'^(holdout_|stress_)', '', run_id)


def load_state(path: Path):
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data.get('queues', {}) if isinstance(data, dict) else {}
    except Exception:
        return {}


def canonical_state_reason(raw: str) -> str:
    if not raw:
        return ""
    return HARDFAIL_MAP.get(raw, str(raw).strip().upper() or "METRICS_MISSING")


queue_root = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
orphan_file = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
run_index_path = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
pre_rank_top_k = int(sys.argv[5]) if len(sys.argv) > 5 and str(sys.argv[5]).strip() else 8
state_path = Path(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else None
state_by_queue = load_state(state_path)

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


def emit_scores():
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
        robust_sharpes = []
        worst_steps = []
        by_state_reason = defaultdict(int)

        state_entry = state_by_queue.get(queue_rel, {})
        state_verdict = str(state_entry.get('promotion_verdict', '') or '').strip().upper()
        state_strict_status = str(state_entry.get('strict_gate_status', '') or '').strip()
        state_strict_reason = str(state_entry.get('strict_gate_reason', '') or '').strip()

        for base, bundle in by_base.items():
            h = bundle.get('holdout')
            s = bundle.get('stress')

            if h is None and s is None:
                unknown_configs += 1
                continue

            h_ok = True
            s_ok = True
            h_reason = ""
            s_reason = ""

            if h is None:
                h_ok = False
                h_reason = "metrics_missing"
            else:
                ok, reason = hard_gate_pass(h)
                if not ok:
                    h_ok = False
                    h_reason = reason

            if s is None:
                s_ok = False
                s_reason = "metrics_missing"
            else:
                ok, reason = hard_gate_pass(s)
                if not ok:
                    s_ok = False
                    s_reason = reason

            if h_ok and s_ok:
                pass_configs += 1

                h_sh = to_float(h.get('sharpe_ratio_abs'))
                s_sh = to_float(s.get('sharpe_ratio_abs'))
                if h_sh is not None and s_sh is not None:
                    robust_sharpes.append(min(h_sh, s_sh))

                h_tail = to_float(h.get('tail_loss_worst_pair_pnl'), None)
                s_tail = to_float(s.get('tail_loss_worst_pair_pnl'), None)
                h_tail2 = to_float(h.get('tail_loss_worst_period_pnl'), None)
                s_tail2 = to_float(s.get('tail_loss_worst_period_pnl'), None)
                tails = [v for v in (h_tail, s_tail, h_tail2, s_tail2) if v is not None]
                if tails:
                    worst_steps.append(min(tails))
            elif (not h_ok) and (not s_ok):
                fail_configs += 1
                reason = HARDFAIL_MAP.get(h_reason, HARDFAIL_MAP.get(s_reason, None))
                if reason:
                    fail_reasons.append(reason)
                    by_state_reason[reason] += 1
            else:
                unknown_configs += 1
                reason = HARDFAIL_MAP.get(h_reason, HARDFAIL_MAP.get(s_reason, None))
                if reason:
                    by_state_reason[reason] += 0.1

        total_configs = pass_configs + fail_configs + unknown_configs

        strict_gate_status = "FULLSPAN_PREFILTER_PASSED"
        strict_gate_reason = ""

        if state_verdict == "REJECT" or state_strict_status == "FULLSPAN_PREFILTER_REJECT":
            promotion_potential = "REJECT"
            gate_status = "HARD_FAIL"
            strict_gate_status = "FULLSPAN_PREFILTER_REJECT"
            gate_reason = canonical_state_reason(state_strict_reason or "METRICS_MISSING")
        elif total_configs == 0:
            promotion_potential = "UNKNOWN"
            gate_status = "OPEN"
            strict_gate_status = "FULLSPAN_PREFILTER_UNKNOWN"
            gate_reason = "insufficient_history"
        elif pass_configs == 0 and fail_configs > 0:
            promotion_potential = "REJECT"
            gate_status = "HARD_FAIL"
            strict_gate_status = "FULLSPAN_PREFILTER_REJECT"
            if fail_reasons:
                gate_reason = fail_reasons[0]
            else:
                # deterministic fallback for full-reject queue
                gate_reason = "METRICS_MISSING"
        else:
            promotion_potential = "POSSIBLE"
            gate_status = "OPEN"
            if fail_configs > 0:
                strict_gate_status = "FULLSPAN_PREFILTER_DEGRADED"
            gate_reason = "history_probe"

        if strict_gate_status in ("FULLSPAN_PREFILTER_REJECT", "HARD_FAIL") and gate_status == "HARD_FAIL":
            strict_gate_reason = gate_reason

        if strict_gate_status in ("FULLSPAN_PREFILTER_REJECT", "HARD_FAIL") and strict_gate_reason:
            strict_gate_reason = canonical_state_reason(strict_gate_reason)

        if gate_status == "OPEN" and strict_gate_status == "FULLSPAN_PREFILTER_PASSED":
            # keep fullspan strict signal as state marker
            strict_gate_reason = ""

        fs_score = score_fullspan_v1_like(robust_sharpes, worst_steps)

        stale_count = failed
        pre_rank_score = 0.0
        if fs_score is not None:
            pre_rank_score += float(fs_score) * 12.0
        else:
            pre_rank_score -= 20.0

        pre_rank_score += pass_configs * 8.0
        pre_rank_score += unknown_configs * 0.5
        pre_rank_score -= fail_configs * 4.5
        pre_rank_score -= stale_count * 1.2
        pre_rank_score -= age_min * 0.3

        pre_rank_score += 20.0 if strict_gate_status == "FULLSPAN_PREFILTER_PASSED" else 0.0

        if state_verdict == "PROMOTE_ELIGIBLE":
            pre_rank_score += 10.0
        elif state_verdict in ("PROMOTE_PENDING_CONFIRM", "PROMOTE_DEFER_CONFIRM"):
            pre_rank_score += 3.0

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
            strict_gate_status,
            strict_gate_reason,
        ))

    out.sort()

    with out_csv.open('w', encoding='utf-8', newline='') as f:
        f.write('queue,planned,running,stalled,failed,completed,total,urgency,mtime,promotion_potential,gate_status,gate_reason,pre_rank_score,strict_gate_status,strict_gate_reason\n')
        if not out:
            raise SystemExit(0)

        out = out[:pre_rank_top_k]
        _, __, ___, queue, planned, running, stalled, failed, completed, total, urgency, potential, gate_status, gate_reason, pre_rank, strict_gate_status, strict_gate_reason = out[0]
        f.write(f"{queue},{planned},{running},{stalled},{failed},{completed},{total},{urgency},{int(Path(queue).stat().st_mtime)},{potential},{gate_status},{gate_reason},{pre_rank},{strict_gate_status},{strict_gate_reason}\n")
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

fullspan_state_set() {
  local queue="$1"
  local verdict="$2"
  local strict_pass_count="${3:-0}"
  local strict_run_groups="${4:-0}"
  local top_run_group="${5:-}"
  local top_variant="${6:-}"
  local top_score="${7:-}"
  local rejection_reason="${8:-}"
  local confirm_count="${9:-0}"
  local strict_gate_status="${10:-FULLSPAN_PREFILTER_UNKNOWN}"
  local strict_gate_reason="${11:-}"
  local run_groups_csv="${12:-}"
  local strict_summary_path="${13:-}"

  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  python3 - "$FULLSPAN_DECISION_STATE_FILE" "$queue" "$verdict" "$strict_pass_count" "$strict_run_groups" "$top_run_group" "$top_variant" "$top_score" "$rejection_reason" "$confirm_count" "$strict_gate_status" "$strict_gate_reason" "$run_groups_csv" "$strict_summary_path" "$ts" <<'PY'
import json
import sys
from pathlib import Path

state_file = Path(sys.argv[1])
queue = sys.argv[2]
verdict = sys.argv[3]
strict_pass_count = sys.argv[4]
strict_run_groups = sys.argv[5]
top_run_group = sys.argv[6]
top_variant = sys.argv[7]
top_score = sys.argv[8]
rejection_reason = sys.argv[9]
confirm_count = sys.argv[10]
strict_gate_status = sys.argv[11]
strict_gate_reason = sys.argv[12]
run_groups_csv = sys.argv[13]
strict_summary_path = sys.argv[14]
ts = sys.argv[15]

state = {}
if state_file.exists():
    try:
        state = json.loads(state_file.read_text(encoding='utf-8'))
    except Exception:
        state = {}
if not isinstance(state, dict):
    state = {}

queues = state.get('queues', {})
if not isinstance(queues, dict):
    queues = {}

run_groups = [item for item in (run_groups_csv.split('||') if run_groups_csv else []) if item]

try:
    strict_pass_count_i = int(float(strict_pass_count or 0))
except Exception:
    strict_pass_count_i = 0
try:
    strict_run_groups_i = int(float(strict_run_groups or 0))
except Exception:
    strict_run_groups_i = 0
try:
    confirm_count_i = int(float(confirm_count or 0))
except Exception:
    confirm_count_i = 0

queues[queue] = {
    "promotion_verdict": verdict,
    "strict_pass_count": strict_pass_count_i,
    "strict_run_group_count": strict_run_groups_i,
    "top_run_group": top_run_group,
    "top_variant": top_variant,
    "top_score": top_score,
    "rejection_reason": rejection_reason,
    "confirm_count": confirm_count_i,
    "selection_policy": "fullspan_v1",
    "selection_mode": "fullspan",
    "strict_gate_status": strict_gate_status,
    "strict_gate_reason": strict_gate_reason,
    "run_groups": run_groups,
    "strict_summary_path": strict_summary_path,
    "last_update": ts,
}
state['queues'] = queues
state['state_version'] = state.get('state_version', 2)
state_file.parent.mkdir(parents=True, exist_ok=True)
state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
PY
}

fullspan_state_get() {
  local queue="$1"
  local field="$2"
  local default_value="${3:-}"
  python3 - "$FULLSPAN_DECISION_STATE_FILE" "$queue" "$field" "$default_value" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
queue = sys.argv[2]
field = sys.argv[3]
default_value = sys.argv[4]
if not path.exists():
    print(default_value)
    raise SystemExit(0)
try:
    data = json.loads(path.read_text(encoding='utf-8'))
    queues = data.get('queues', {})
    if isinstance(queues, dict):
        print(queues.get(queue, {}).get(field, default_value))
        raise SystemExit(0)
except Exception:
    pass
print(default_value)
PY
}

fullspan_state_metric_get() {
  local metric="$1"
  local default_value="${2:-0}"
  python3 - "$FULLSPAN_DECISION_STATE_FILE" "$metric" "$default_value" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metric = sys.argv[2]
default_value = sys.argv[3]
if not path.exists():
    print(default_value)
    raise SystemExit(0)
try:
    data = json.loads(path.read_text(encoding='utf-8'))
    metrics = data.get('runtime_metrics', {})
    if isinstance(metrics, dict):
        print(metrics.get(metric, default_value))
        raise SystemExit(0)
except Exception:
    pass
print(default_value)
PY
}

fullspan_state_metric_inc() {
  local metric="$1"
  local delta="${2:-1}"
  python3 - "$FULLSPAN_DECISION_STATE_FILE" "$metric" "$delta" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metric = sys.argv[2]
delta = sys.argv[3]
state = {}
if path.exists():
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        state = {}
if not isinstance(state, dict):
    state = {}
metrics = state.get('runtime_metrics', {})
if not isinstance(metrics, dict):
    metrics = {}
try:
    d = int(float(delta))
except Exception:
    d = 0
try:
    metrics[metric] = int(metrics.get(metric, 0)) + d
except Exception:
    metrics[metric] = 0
state['runtime_metrics'] = metrics
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
PY
}

fullspan_state_metric_set() {
  local metric="$1"
  local value="$2"
  python3 - "$FULLSPAN_DECISION_STATE_FILE" "$metric" "$value" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metric = sys.argv[2]
value = sys.argv[3]
state = {}
if path.exists():
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        state = {}
if not isinstance(state, dict):
    state = {}
metrics = state.get('runtime_metrics', {})
if not isinstance(metrics, dict):
    metrics = {}
try:
    metrics[metric] = int(value)
except Exception:
    metrics[metric] = value
state['runtime_metrics'] = metrics
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
PY
}

fullspan_cycle_cache_get() {
  local queue="$1"
  local fingerprint="$2"
  local summary_path="$3"
  python3 - "$FULLSPAN_CYCLE_CACHE_FILE" "$queue" "$fingerprint" "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
queue = sys.argv[2]
fingerprint = sys.argv[3]
summary_path = Path(sys.argv[4])
if not path.exists():
    raise SystemExit(1)
try:
    data = json.loads(path.read_text(encoding='utf-8'))
except Exception:
    raise SystemExit(1)
queues = data.get('queues', {})
if not isinstance(queues, dict):
    raise SystemExit(1)
entry = queues.get(queue)
if not isinstance(entry, dict):
    raise SystemExit(1)
if str(entry.get('fingerprint', '')) != str(fingerprint):
    raise SystemExit(1)
if not summary_path.exists():
    raise SystemExit(1)
print('1')
PY
}

fullspan_cycle_cache_set() {
  local queue="$1"
  local fingerprint="$2"
  local strict_pass_count="$3"
  local strict_run_group_count="$4"
  local summary_path="$5"
  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  python3 - "$FULLSPAN_CYCLE_CACHE_FILE" "$queue" "$fingerprint" "$strict_pass_count" "$strict_run_group_count" "$summary_path" "$ts" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
queue = sys.argv[2]
fingerprint = sys.argv[3]
try:
    strict_pass_count = int(float(sys.argv[4]))
except Exception:
    strict_pass_count = 0
try:
    strict_run_group_count = int(float(sys.argv[5]))
except Exception:
    strict_run_group_count = 0
summary_path = sys.argv[6]
ts = sys.argv[7]
state = {}
if path.exists():
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        state = {}
if not isinstance(state, dict):
    state = {}
queues = state.get('queues', {})
if not isinstance(queues, dict):
    queues = {}
queues[queue] = {
    'fingerprint': str(fingerprint),
    'strict_pass_count': strict_pass_count,
    'strict_run_group_count': strict_run_group_count,
    'summary_path': summary_path,
    'last_update': ts,
}
state['queues'] = queues
state['version'] = state.get('version', 1)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
PY
}


fullspan_confirm_count_for_queue() {
  local top_run_group="$1"
  local top_variant="$2"

  if [[ -z "$top_run_group" && -z "$top_variant" ]]; then
    echo 0
    return 0
  fi

  if [[ ! -f "$RUN_INDEX_PATH" ]]; then
    echo 0
    return 0
  fi

  python3 - "$RUN_INDEX_PATH" "$top_run_group" "$top_variant" <<'PY'
import csv
import sys
from pathlib import Path

run_index_path = Path(sys.argv[1])
top_run_group = (sys.argv[2] or "").strip()
top_variant = (sys.argv[3] or "").strip()

if not run_index_path.exists():
    print(0)
    raise SystemExit(0)

run_groups = set()
try:
    with run_index_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            status = (row.get("status") or "").strip().lower()
            if status != "completed":
                continue

            run_group = (row.get("run_group") or "").strip()
            run_id = (row.get("run_id") or "").strip()
            config_path = (row.get("config_path") or "").strip().lower()
            results_dir = (row.get("results_dir") or "").strip().lower()

            variant_match = bool(top_variant) and (
                top_variant.lower() in run_id.lower() or top_variant.lower() in config_path
            )
            group_match = bool(top_run_group) and (
                top_run_group in run_group or top_run_group in run_id
            )
            if not (variant_match or group_match):
                continue

            run_dir_hint = "confirm" in (run_group + "/" + run_id + "/" + results_dir).lower()
            if not run_dir_hint:
                continue

            if run_group:
                run_groups.add(run_group)

    print(len(run_groups))
except Exception:
    print(0)
PY
}

run_fullspan_cycle() {
  local queue_rel="$1"
  local queue_path="$2"
  local queue_name="$3"

  if [[ -z "$queue_rel" || -z "$queue_path" ]]; then
    return 0
  fi
  if [[ ! -f "$queue_path" ]]; then
    return 0
  fi

  local safe_name
  safe_name="$(printf '%s' "$queue_rel" | tr '/.' '__')"
  local cycle_log="$STATE_DIR/fullspan_cycle_${safe_name}.log"
  local cycle_summary="$STATE_DIR/fullspan_cycle_${safe_name}.json"
  local now_epoch
  now_epoch="$(date +%s)"

  local run_index_sig="none"
  if [[ -f "$RUN_INDEX_PATH" ]]; then
    run_index_sig="$(sha256sum "$RUN_INDEX_PATH" | awk '{print $1}')"
  fi

  local queue_sig
  queue_sig="$(python3 - "$queue_path" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
except Exception:
    rows = []
h = hashlib.sha256()
h.update(json.dumps(rows, sort_keys=True).encode('utf-8'))
print(h.hexdigest())
PY
)"

  local decision_fingerprint
  decision_fingerprint="${queue_sig}|${run_index_sig}|${PROMOTION_SELECTION_PROFILE}|${PROMOTION_SELECTION_MODE}|${FULLSPAN_POLICY_NAME}|${PROMOTION_PRE_RANK_TOPK}|${FULLSPAN_CONFIRM_MIN_GROUPS}|${FULLSPAN_CONFIRM_MIN_REPLIES}"

  if [[ -s "$cycle_summary" ]] && fullspan_cycle_cache_get "$queue_rel" "$decision_fingerprint" "$cycle_summary" >/dev/null 2>&1; then
    log "decision_cycle_cached queue=$queue_rel fingerprint=$decision_fingerprint"
    parsed="$(python3 - "$cycle_summary" - <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
strict_data = data.get('strict', {})
run_groups = strict_data.get('run_groups', []) or []
print(int(strict_data.get('pass_count', 0) or 0))
print(int(strict_data.get('run_group_count', 0) or 0))
print('||'.join(run_groups))
print(strict_data.get('top_run_group', '') or '')
print(strict_data.get('top_variant', '') or '')
print(strict_data.get('top_score', '') or '')
print((strict_data.get('rejection_reason_line', '') or '').replace('\n', ' '))
PY
)"

    if [[ -n "$parsed" ]]; then
      readarray -t parsed_lines <<< "$parsed"
      strict_pass_count="${parsed_lines[0]:-0}"
      strict_run_group_count="${parsed_lines[1]:-0}"
      strict_run_groups="${parsed_lines[2]:-}"
      top_run_group="${parsed_lines[3]:-}"
      top_variant="${parsed_lines[4]:-}"
      top_score="${parsed_lines[5]:-}"
      strict_rejection_reason="${parsed_lines[6]:-}"

      local confirm_count
      confirm_count="0"
      if [[ "$strict_pass_count" -gt 0 ]]; then
        confirm_count="$(fullspan_confirm_count_for_queue "$top_run_group" "$top_variant")"
      fi

      if [[ "$strict_pass_count" -le 0 ]]; then
        fullspan_state_set "$queue_rel" "REJECT" \
          "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
          "${strict_rejection_reason:-METRICS_MISSING}" "$confirm_count" "FULLSPAN_PREFILTER_REJECT" "" "$strict_run_groups" "$cycle_summary"
      elif (( strict_run_group_count >= FULLSPAN_CONFIRM_MIN_GROUPS )); then
        if (( confirm_count >= FULLSPAN_CONFIRM_MIN_REPLIES )); then
          fullspan_state_set "$queue_rel" "PROMOTE_ELIGIBLE" \
            "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
            "" "$confirm_count" "FULLSPAN_PREFILTER_PASSED" "" "$strict_run_groups" "$cycle_summary"
        else
          fullspan_state_set "$queue_rel" "PROMOTE_PENDING_CONFIRM" \
            "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
            "pending_confirm" "$confirm_count" "FULLSPAN_PREFILTER_PASSED" "" "$strict_run_groups" "$cycle_summary"
        fi
      else
        fullspan_state_set "$queue_rel" "PROMOTE_DEFER_CONFIRM" \
          "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
          "insufficient_run_groups" "$confirm_count" "FULLSPAN_PREFILTER_PASSED" "" "$strict_run_groups" "$cycle_summary"
      fi

      fullspan_cycle_cache_set "$queue_rel" "$decision_fingerprint" "$strict_pass_count" "$strict_run_group_count" "$cycle_summary"
      printf '%s\n' "$queue_rel" >> "$FULLSPAN_CYCLE_STATE_FILE"
      return 0
    fi
  fi

  log "decision_cycle_start queue=$queue_rel cycle_log=$cycle_log fingerprint=$decision_fingerprint"

  fullspan_state_metric_inc "fullspan_cycle_counter" 1
  local cycle_count
  cycle_count="$(fullspan_state_metric_get fullspan_cycle_counter 0)"

  (cd "$ROOT_DIR" && ./.venv/bin/python scripts/optimization/run_fullspan_decision_cycle.py --queue "$queue_rel" --contains "$queue_name" --summary-json "$cycle_summary" >> "$cycle_log" 2>&1)
  local cycle_rc=$?

  strict_pass_count=0
  strict_run_group_count=0
  strict_run_groups=""
  top_run_group=""
  top_variant=""
  top_score=""
  strict_rejection_reason=""

  if [[ -f "$cycle_summary" ]]; then
    parsed="$(python3 - "$cycle_summary" - <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
strict_data = data.get('strict', {})
run_groups = strict_data.get('run_groups', []) or []
print(int(strict_data.get('pass_count', 0) or 0))
print(int(strict_data.get('run_group_count', 0) or 0))
print('||'.join(run_groups))
print(strict_data.get('top_run_group', '') or '')
print(strict_data.get('top_variant', '') or '')
print(strict_data.get('top_score', '') or '')
print((strict_data.get('rejection_reason_line', '') or '').replace('\n', ' '))
PY
)"
    readarray -t parsed_lines <<< "$parsed"
    strict_pass_count="${parsed_lines[0]:-0}"
    strict_run_group_count="${parsed_lines[1]:-0}"
    strict_run_groups="${parsed_lines[2]:-}"
    top_run_group="${parsed_lines[3]:-}"
    top_variant="${parsed_lines[4]:-}"
    top_score="${parsed_lines[5]:-}"
    strict_rejection_reason="${parsed_lines[6]:-}"
  fi

  local confirm_count
  confirm_count="0"
  if [[ "$strict_pass_count" -gt 0 ]]; then
    confirm_count="$(fullspan_confirm_count_for_queue "$top_run_group" "$top_variant")"
  fi

  if (( cycle_rc == 0 )); then
    if (( strict_pass_count <= 0 )); then
      fullspan_state_set "$queue_rel" "REJECT" \
        "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
        "${strict_rejection_reason:-METRICS_MISSING}" "$confirm_count" "FULLSPAN_PREFILTER_REJECT" "" "$strict_run_groups" "$cycle_summary"
      fullspan_state_metric_inc "strict_fullspan_reject_count" 1
      fullspan_state_metric_set "cycles_since_last_strict_pass" "$cycle_count"
      fullspan_state_metric_set "last_decision_epoch" "$now_epoch"
      log_decision_note "$queue_rel" "FULLSPAN_REJECT" "no_strict_fullspan_pass ${strict_rejection_reason}" "await_fullspan_improvement"
    else
      fullspan_state_metric_inc "strict_fullspan_pass_count" 1
      local prev_pass_cycle
      prev_pass_cycle="$(fullspan_state_metric_get last_strict_pass_cycle 0)"
      if [[ -n "$prev_pass_cycle" && "$prev_pass_cycle" != "0" ]]; then
        if awk -v a="$prev_pass_cycle" -v b="$cycle_count" 'BEGIN {if (b>=a) exit 0; else exit 1}'; then
          fullspan_state_metric_set "cycles_to_last_strict_pass" "$((cycle_count - prev_pass_cycle))"
        fi
      fi
      fullspan_state_metric_set "last_strict_pass_cycle" "$cycle_count"
      fullspan_state_metric_set "cycles_since_last_strict_pass" "0"
      fullspan_state_metric_set "last_strict_pass_epoch" "$now_epoch"
      fullspan_state_metric_set "last_decision_epoch" "$now_epoch"

      if (( strict_run_group_count >= FULLSPAN_CONFIRM_MIN_GROUPS )); then
        if (( confirm_count >= FULLSPAN_CONFIRM_MIN_REPLIES )); then
          fullspan_state_set "$queue_rel" "PROMOTE_ELIGIBLE" \
            "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
            "" "$confirm_count" "FULLSPAN_PREFILTER_PASSED" "" "$strict_run_groups" "$cycle_summary"
          fullspan_state_metric_set "promotion_eligible_count" "$(( $(fullspan_state_metric_get promotion_eligible_count 0) + 1 ))"
          log_decision_note "$queue_rel" "FULLSPAN_STRICT_PROMOTE_ELIGIBLE" "strict_pass_and_min_run_groups+confirm" "candidate_still_queued_or_candidate_or_waiting"
        else
          fullspan_state_set "$queue_rel" "PROMOTE_PENDING_CONFIRM" \
            "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
            "pending_confirm" "$confirm_count" "FULLSPAN_PREFILTER_PASSED" "" "$strict_run_groups" "$cycle_summary"
          fullspan_state_metric_set "confirm_pending_count" "$(( $(fullspan_state_metric_get confirm_pending_count 0) + 1 ))"
          log_decision_note "$queue_rel" "FULLSPAN_STRICT_PENDING_CONFIRM" "strict_pass_run_groups=$strict_run_group_count confirm_count=$confirm_count" "need_vps_confirm_replay"
        fi
      else
        fullspan_state_set "$queue_rel" "PROMOTE_DEFER_CONFIRM" \
          "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
          "insufficient_run_groups" "$confirm_count" "FULLSPAN_PREFILTER_PASSED" "" "$strict_run_groups" "$cycle_summary"
        fullspan_state_metric_set "confirm_pending_count" "$(( $(fullspan_state_metric_get confirm_pending_count 0) + 1 ))"
        log_decision_note "$queue_rel" "FULLSPAN_STRICT_PASS" "strict_pass_count=$strict_pass_count run_groups=$strict_run_group_count" "need_additional_run_groups"
      fi
    fi

    fullspan_cycle_cache_set "$queue_rel" "$decision_fingerprint" "$strict_pass_count" "$strict_run_group_count" "$cycle_summary"
    printf '%s\n' "$queue_rel" >> "$FULLSPAN_CYCLE_STATE_FILE"
  else
    log "decision_cycle_fail queue=$queue_rel rc=$cycle_rc"
    fullspan_state_metric_set "decision_cycle_fail_count" "$(( $(fullspan_state_metric_get decision_cycle_fail_count 0) + 1 ))"
  fi

  return 0
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
declare -A no_progress_streak_by_queue

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

  IFS=',' read -r queue planned running stalled failed completed total urgency mtime promotion_potential gate_status gate_reason pre_rank_score strict_gate_status strict_gate_reason < <(tail -n 1 "$CANDIDATE_FILE")

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
  state_verdict="$(fullspan_state_get "$queue_rel" "promotion_verdict" "ANALYZE")"
  state_rejection_reason="$(fullspan_state_get "$queue_rel" "rejection_reason" "")"
  state_strict_pass_count="$(fullspan_state_get "$queue_rel" "strict_pass_count" "0")"
  state_strict_gate_status="$(fullspan_state_get "$queue_rel" "strict_gate_status" "FULLSPAN_PREFILTER_UNKNOWN")"
  state_strict_gate_reason="$(fullspan_state_get "$queue_rel" "strict_gate_reason" "")"
  state_strict_run_groups="$(fullspan_state_get "$queue_rel" "strict_run_group_count" "0")"
  state_confirm_count="$(fullspan_state_get "$queue_rel" "confirm_count" "0")"

  if [[ "$state_verdict" == "REJECT" || "$state_strict_gate_status" == "FULLSPAN_PREFILTER_REJECT" ]]; then
    promotion_verdict="REJECT"
    promotion_potential="REJECT"
    gate_status="HARD_FAIL"
    gate_reason="${state_strict_gate_reason:-${state_rejection_reason:-state_reject}}"
  elif [[ "$state_verdict" == "PROMOTE_ELIGIBLE" ]]; then
    promotion_verdict="PROMOTE_ELIGIBLE"
    gate_status="OPEN"
  elif [[ "$state_verdict" == "PROMOTE_PENDING_CONFIRM" || "$state_verdict" == "PROMOTE_DEFER_CONFIRM" ]]; then
    promotion_verdict="DEFER_CONFIRM"
  fi

  if [[ "$promotion_potential" == "REJECT" && "$gate_status" == "HARD_FAIL" ]]; then
    promotion_verdict="REJECT"
  fi

  if [[ -n "$prev_queue" && "$prev_queue" == "$queue_rel" ]]; then
    if [[ "$pending" -lt "$prev_pending" ]]; then
      clear_orphan "$queue_rel"
      no_progress_streak_by_queue["$queue_rel"]=0
      log "progress_seen queue=$queue_rel prev=$prev_pending curr=$pending"
      sync_queue_status "$queue_rel"
    fi
  fi

  heartbeat_update "$queue_rel" "$pending" "$completed" "$total" "$planned" "$running" "$stalled"

  if (( pending > 0 )); then
    no_progress_count="${no_progress_streak_by_queue[$queue_rel]:-0}"
    if [[ -n "$prev_queue" && "$prev_queue" == "$queue_rel" ]]; then
      if [[ "$pending" -eq "$prev_pending" && "$running" -eq "$prev_running" && "$stalled" -eq "$prev_stalled" && "$planned" -eq "$prev_planned" && "$failed" -eq "$prev_failed" ]]; then
        no_progress_count=$((no_progress_count + 1))
      else
        no_progress_count=0
      fi
    else
      no_progress_count=0
    fi

    no_progress_streak_by_queue["$queue_rel"]="$no_progress_count"
    if (( no_progress_count >= NO_PROGRESS_STALE_CYCLES )); then
      mark_orphan "$queue_rel" "no_progress_streak_${no_progress_count}"
      fullspan_state_set "$queue_rel" "REJECT" "$state_strict_pass_count" "$state_strict_run_groups" "" "" "" "no_progress_streak" "$state_confirm_count" "FULLSPAN_PREFILTER_REJECT"
      log_decision_note "$queue_rel" "FULLSPAN_NO_PROGRESS_FAIL_CLOSED" "no_progress_streak=${no_progress_count}" "retry_orphan_closed"
      log "no_progress_fail_closed queue=$queue_rel streak=$no_progress_count pending=$pending"
      sleep 2
      continue
    fi
  else
    no_progress_streak_by_queue["$queue_rel"]=0
  fi

  safe_queue_name="$(basename "$queue")"
  if [[ "$pending" -eq 0 && "$completed" -gt 0 ]]; then
    run_fullspan_cycle "$queue_rel" "$queue" "$safe_queue_name"
  fi

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
    if [[ "$completed" -gt 0 ]]; then
      safe_queue_name="$(basename "$queue")"
      run_fullspan_cycle "$queue_rel" "$queue" "$safe_queue_name"
    fi
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
