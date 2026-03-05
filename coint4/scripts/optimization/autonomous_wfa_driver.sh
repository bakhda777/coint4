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
POWEROFF_AFTER_RUN="${POWEROFF_AFTER_RUN:-true}"
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
DECISION_MEMO_DIR="$STATE_DIR/decision_memos"
DETERMINISTIC_QUARANTINE_FILE="$STATE_DIR/deterministic_quarantine.json"


FULLSPAN_CYCLE_STATE_FILE="$STATE_DIR/mini_cycle_state.txt"
FULLSPAN_CYCLE_CACHE_FILE="$STATE_DIR/fullspan_cycle_cache.json"
FULLSPAN_DECISION_STATE_FILE="$STATE_DIR/fullspan_decision_state.json"
FULLSPAN_CONFIRM_MIN_GROUPS="${FULLSPAN_CONFIRM_MIN_GROUPS:-2}"
FULLSPAN_CONFIRM_MIN_REPLIES="${FULLSPAN_CONFIRM_MIN_REPLIES:-2}"
FULLSPAN_ROLLUP_SYNC_MIN_INTERVAL="${FULLSPAN_ROLLUP_SYNC_MIN_INTERVAL:-60}"
FULLSPAN_ROLLUP_SYNC_MARKER="$STATE_DIR/fullspan_rollup_sync.marker"
NO_PROGRESS_STALE_CYCLES="${NO_PROGRESS_STALE_CYCLES:-6}"
SAME_REASON_REPAIR_CAP="${SAME_REASON_REPAIR_CAP:-3}"
ADAPTIVE_LOW_RATE_THRESHOLD="${ADAPTIVE_LOW_RATE_THRESHOLD:-0.05}"
ADAPTIVE_HIGH_RATE_THRESHOLD="${ADAPTIVE_HIGH_RATE_THRESHOLD:-0.60}"
SLA_VPS_IDLE_PENDING_CYCLES="${SLA_VPS_IDLE_PENDING_CYCLES:-3}"
SLA_CONFIRM_PENDING_SEC="${SLA_CONFIRM_PENDING_SEC:-7200}"
CONFIRM_FASTLANE_LIMIT="${CONFIRM_FASTLANE_LIMIT:-1}"
CONFIRM_FASTLANE_PARALLEL="${CONFIRM_FASTLANE_PARALLEL:-2}"
CONFIRM_FASTLANE_COOLDOWN_SEC="${CONFIRM_FASTLANE_COOLDOWN_SEC:-1800}"
LOW_YIELD_HARDFAIL_STREAK_LIMIT="${LOW_YIELD_HARDFAIL_STREAK_LIMIT:-3}"
LOW_YIELD_HOMOGENEOUS_FAIL_FRACTION="${LOW_YIELD_HOMOGENEOUS_FAIL_FRACTION:-0.70}"
LOW_YIELD_HOMOGENEOUS_FAIL_MIN="${LOW_YIELD_HOMOGENEOUS_FAIL_MIN:-2}"

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

  python3 - "$QUEUE_ROOT" "$CANDIDATE_FILE" "$orphan_path" "$RUN_INDEX_PATH" "$PROMOTION_PRE_RANK_TOPK" "$FULLSPAN_DECISION_STATE_FILE" "$LOW_YIELD_HOMOGENEOUS_FAIL_FRACTION" "$LOW_YIELD_HOMOGENEOUS_FAIL_MIN" "$FULLSPAN_CONFIRM_MIN_GROUPS" <<'PY'
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
    if worst_dd > 0.20:
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


LOW_YIELD_HOMOGENEOUS_FRACTION = 0.7
try:
    LOW_YIELD_HOMOGENEOUS_FRACTION = float(sys.argv[7])
except Exception:
    LOW_YIELD_HOMOGENEOUS_FRACTION = 0.7

try:
    LOW_YIELD_HOMOGENEOUS_MIN = int(sys.argv[8])
except Exception:
    LOW_YIELD_HOMOGENEOUS_MIN = 2

try:
    FULLSPAN_CONFIRM_MIN_GROUPS = int(sys.argv[9])
except Exception:
    FULLSPAN_CONFIRM_MIN_GROUPS = 2


queue_root = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
try:
    app_root = queue_root.parents[2]
except Exception:
    app_root = queue_root
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

        queue_abs = str(p)
        try:
            queue_rel = str(p.relative_to(app_root))
        except Exception:
            queue_rel = queue_abs

        orphan_until = None
        for k in (queue_rel, queue_abs, '/' + queue_rel.lstrip('/')):
            if k in orphan:
                orphan_until = orphan.get(k, 0.0)
                break
        if orphan_until is not None:
            try:
                if now < float(orphan_until or 0.0):
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
        if not state_entry and queue_abs in state_by_queue:
            state_entry = state_by_queue.get(queue_abs, {})
        state_verdict = str(state_entry.get('promotion_verdict', '') or '').strip().upper()
        state_strict_status = str(state_entry.get('strict_gate_status', '') or '').strip()
        state_strict_reason = str(state_entry.get('strict_gate_reason', '') or '').strip()
        try:
            state_strict_pass_count = int(float(state_entry.get('strict_pass_count', 0) or 0))
        except Exception:
            state_strict_pass_count = 0
        try:
            state_strict_run_groups = int(float(state_entry.get('strict_run_group_count', 0) or 0))
        except Exception:
            state_strict_run_groups = 0
        try:
            state_confirm_count = int(float(state_entry.get('confirm_count', 0) or 0))
        except Exception:
            state_confirm_count = 0
        try:
            confirm_pending_since_epoch = int(float(state_entry.get('confirm_pending_since_epoch', 0) or 0))
        except Exception:
            confirm_pending_since_epoch = 0

        confirm_fastlane_ready = bool(
            state_verdict in ("PROMOTE_PENDING_CONFIRM", "PROMOTE_DEFER_CONFIRM")
            and state_strict_pass_count > 0
            and state_strict_run_groups >= FULLSPAN_CONFIRM_MIN_GROUPS
        )

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
                dominant_reason = fail_reasons[0]
            else:
                dominant_reason = ""

            if by_state_reason and fail_configs >= LOW_YIELD_HOMOGENEOUS_MIN:
                try:
                    dominant_reason_name, dominant_reason_count = max(by_state_reason.items(), key=lambda item: item[1])
                    threshold = max(LOW_YIELD_HOMOGENEOUS_MIN, int(fail_configs * LOW_YIELD_HOMOGENEOUS_FRACTION))
                    if dominant_reason_count >= threshold:
                        dominant_reason = f"LOW_YIELD_HOMOGENEOUS_{dominant_reason_name}"
                except Exception:
                    pass

            if dominant_reason:
                gate_reason = dominant_reason
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

        # distance-to-goal prioritization: prioritize queues closer to strict-pass + confirm completion
        pre_rank_score += min(max(state_strict_pass_count, 0), 3) * 6.0
        pre_rank_score += min(max(state_strict_run_groups, 0), 2) * 10.0
        pre_rank_score += min(max(state_confirm_count, 0), 2) * 12.0

        if state_verdict == "PROMOTE_ELIGIBLE":
            pre_rank_score += 10.0
        elif confirm_fastlane_ready:
            pre_rank_score += 20.0
            if confirm_pending_since_epoch > 0:
                age_hours = max(0.0, (now - confirm_pending_since_epoch) / 3600.0)
                pre_rank_score += min(age_hours, 24.0) * 1.5

        if fail_configs > 0 and by_state_reason:
            dominant_reason_count = max(by_state_reason.values())
            if dominant_reason_count >= max(2, int(0.7 * max(fail_configs, 1))):
                # repeated same fail-pattern likely low-yield queue; de-prioritize
                pre_rank_score -= 6.0

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
            int(mtime),
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
        _, __, ___, queue, planned, running, stalled, failed, completed, total, urgency, mtime_i, potential, gate_status, gate_reason, pre_rank, strict_gate_status, strict_gate_reason = out[0]
        f.write(f"{queue},{planned},{running},{stalled},{failed},{completed},{total},{urgency},{mtime_i},{potential},{gate_status},{gate_reason},{pre_rank},{strict_gate_status},{strict_gate_reason}\n")
PY
}

fallback_pending_candidate() {
  python3 - "$QUEUE_ROOT" "$CANDIDATE_FILE" <<'PY'
import csv
from collections import Counter
from pathlib import Path
import time
import sys

queue_root = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
try:
    app_root = queue_root.parents[2]
except Exception:
    app_root = queue_root

best = None
for p in sorted(queue_root.rglob('run_queue.csv')):
    try:
        rows = list(csv.DictReader(p.open(newline='', encoding='utf-8')))
    except Exception:
        continue
    c = Counter((r.get('status') or '').strip().lower() for r in rows)
    planned = int(c.get('planned', 0))
    running = int(c.get('running', 0))
    stalled = int(c.get('stalled', 0))
    failed = int(c.get('failed', 0)) + int(c.get('error', 0))
    completed = int(c.get('completed', 0))
    total = len(rows)
    pending = planned + running + stalled + failed
    if pending <= 0 or total <= 0:
        continue
    age_min = max(0.0, (time.time() - p.stat().st_mtime) / 60.0)
    urgency = (stalled * 100.0) + (running * 20.0) + (age_min * 0.1) + 1.0
    row = (pending, stalled, running, -int(p.stat().st_mtime), str(p), planned, running, stalled, failed, completed, total, urgency)
    if best is None or row > best:
        best = row

if best is None:
    raise SystemExit(1)

_, _, _, _, queue, planned, running, stalled, failed, completed, total, urgency = best
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open('w', encoding='utf-8', newline='') as f:
    f.write('queue,planned,running,stalled,failed,completed,total,urgency,mtime,promotion_potential,gate_status,gate_reason,pre_rank_score,strict_gate_status,strict_gate_reason\n')
    f.write(f"{queue},{planned},{running},{stalled},{failed},{completed},{total},{urgency:.3f},{int(Path(queue).stat().st_mtime)},POSSIBLE,OPEN,fallback_pending,0.000,FULLSPAN_PREFILTER_UNKNOWN,\n")
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
  load1="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=6 "$SERVER_USER@$SERVER_IP" "cat /proc/loadavg 2>/dev/null | awk '{print \$1}'" || true)"
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

  # Adaptive parallel by observed progress rate (from heartbeat).
  local rate
  rate="${hb_rate_per_min:-0}"
  if awk -v v="$rate" -v th="$ADAPTIVE_LOW_RATE_THRESHOLD" 'BEGIN { exit (v+0 <= th+0) ? 0 : 1 }'; then
    p=$((p - 1))
  elif awk -v v="$rate" -v th="$ADAPTIVE_HIGH_RATE_THRESHOLD" 'BEGIN { exit (v+0 >= th+0) ? 0 : 1 }'; then
    p=$((p + 1))
  fi

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

write_decision_memo() {
  local queue_rel="$1"
  local verdict="$2"
  local strict_pass_count="$3"
  local strict_run_group_count="$4"
  local confirm_count="$5"
  local rejection_reason="$6"
  local top_variant="$7"
  local top_run_group="$8"
  local top_score="$9"
  local strict_gate_status="${10:-}"
  local strict_gate_reason="${11:-}"

  python3 - "$DECISION_MEMO_DIR" "$queue_rel" "$verdict" "$strict_pass_count" "$strict_run_group_count" "$confirm_count" "$rejection_reason" "$top_variant" "$top_run_group" "$top_score" "$strict_gate_status" "$strict_gate_reason" "$FULLSPAN_CONFIRM_MIN_GROUPS" "$FULLSPAN_CONFIRM_MIN_REPLIES" <<'PY'
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

memo_dir = Path(sys.argv[1])
queue_rel = sys.argv[2]
verdict = sys.argv[3]
strict_pass_count = int(float(sys.argv[4] or 0))
strict_run_group_count = int(float(sys.argv[5] or 0))
confirm_count = int(float(sys.argv[6] or 0))
rejection_reason = sys.argv[7]
top_variant = sys.argv[8]
top_run_group = sys.argv[9]
top_score = sys.argv[10]
strict_gate_status = sys.argv[11]
strict_gate_reason = sys.argv[12]
confirm_min_groups = int(float(sys.argv[13] or 2))
confirm_min_replies = int(float(sys.argv[14] or 2))

missing = []
if strict_pass_count <= 0:
    missing.append("strict_pass_by_fullspan")
if strict_run_group_count < confirm_min_groups:
    missing.append(f"min_run_groups:{confirm_min_groups}")
if confirm_count < confirm_min_replies:
    missing.append(f"confirm_replays:{confirm_min_replies}")
if verdict != "PROMOTE_ELIGIBLE":
    missing.append("promotion_not_eligible")

payload = {
    "ts": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    "queue": queue_rel,
    "promotion_verdict": verdict,
    "strict_pass_count": strict_pass_count,
    "strict_run_group_count": strict_run_group_count,
    "confirm_count": confirm_count,
    "rejection_reason": rejection_reason,
    "strict_gate_status": strict_gate_status,
    "strict_gate_reason": strict_gate_reason,
    "top_variant": top_variant,
    "top_run_group": top_run_group,
    "top_score": top_score,
    "missing_for_promote": missing,
}

memo_dir.mkdir(parents=True, exist_ok=True)
safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", queue_rel)
(memo_dir / f"{safe}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
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

fullspan_state_queue_set() {
  local queue="$1"
  shift
  if (( $# % 2 != 0 )); then
    return 0
  fi

  python3 - "$FULLSPAN_DECISION_STATE_FILE" "$queue" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
queue = sys.argv[2]
updates = {}
for i in range(3, len(sys.argv), 2):
    key = sys.argv[i - 1]
    val = sys.argv[i]
    updates[key] = val

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

entry = queues.get(queue, {})
if not isinstance(entry, dict):
    entry = {}

for key, val in updates.items():
    if isinstance(val, str):
        sv = val.strip()
        if sv == "":
            entry[key] = sv
            continue
        try:
            if any(ch in sv for ch in '.eE'):
                fv = float(sv)
                entry[key] = int(fv) if fv.is_integer() else fv
            else:
                entry[key] = int(sv)
            continue
        except Exception:
            pass
    entry[key] = val

queues[queue] = entry
state['queues'] = queues
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
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

fullspan_state_run_groups_csv() {
  local queue="$1"
  python3 - "$FULLSPAN_DECISION_STATE_FILE" "$queue" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
queue = sys.argv[2]
if not path.exists():
    print("")
    raise SystemExit(0)

try:
    data = json.loads(path.read_text(encoding="utf-8"))
    queues = data.get("queues", {})
    entry = queues.get(queue, {}) if isinstance(queues, dict) else {}
    raw = entry.get("run_groups", []) if isinstance(entry, dict) else []
    if isinstance(raw, list):
        print("||".join(str(item) for item in raw if str(item).strip()))
        raise SystemExit(0)
except Exception:
    pass
print("")
PY
}

fullspan_reconcile_confirm_progress() {
  local queue_rel="$1"

  local state_verdict
  local strict_run_group_count
  local strict_pass_count
  local top_run_group
  local top_variant
  local top_score
  local rejection_reason
  local strict_gate_status
  local strict_gate_reason
  local strict_summary_path
  local state_confirm_count

  state_verdict="$(fullspan_state_get "$queue_rel" "promotion_verdict" "ANALYZE")"
  if [[ "$state_verdict" != "PROMOTE_PENDING_CONFIRM" && "$state_verdict" != "PROMOTE_DEFER_CONFIRM" ]]; then
    return 0
  fi

  strict_run_group_count="$(fullspan_state_get "$queue_rel" "strict_run_group_count" "0")"
  strict_pass_count="$(fullspan_state_get "$queue_rel" "strict_pass_count" "0")"
  top_run_group="$(fullspan_state_get "$queue_rel" "top_run_group" "")"
  top_variant="$(fullspan_state_get "$queue_rel" "top_variant" "")"
  top_score="$(fullspan_state_get "$queue_rel" "top_score" "")"
  rejection_reason="$(fullspan_state_get "$queue_rel" "rejection_reason" "")"
  strict_gate_status="$(fullspan_state_get "$queue_rel" "strict_gate_status" "FULLSPAN_PREFILTER_UNKNOWN")"
  strict_gate_reason="$(fullspan_state_get "$queue_rel" "strict_gate_reason" "")"
  strict_summary_path="$(fullspan_state_get "$queue_rel" "strict_summary_path" "")"
  state_confirm_count="$(fullspan_state_get "$queue_rel" "confirm_count" "0")"

  if [[ "$strict_run_group_count" -lt "$FULLSPAN_CONFIRM_MIN_GROUPS" ]]; then
    return 0
  fi

  if [[ -z "$top_run_group" && -z "$top_variant" ]]; then
    return 0
  fi

  local live_confirm_count
  live_confirm_count="$(fullspan_confirm_count_for_queue "$top_run_group" "$top_variant")"

  local run_groups_csv
  run_groups_csv="$(fullspan_state_run_groups_csv "$queue_rel")"

  if (( live_confirm_count >= FULLSPAN_CONFIRM_MIN_REPLIES )); then
    if [[ "$state_verdict" == "PROMOTE_PENDING_CONFIRM" || "$state_verdict" == "PROMOTE_DEFER_CONFIRM" ]]; then
      fullspan_state_set "$queue_rel" "PROMOTE_ELIGIBLE" \
        "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
        "" "$live_confirm_count" "$strict_gate_status" "$strict_gate_reason" "$run_groups_csv" "$strict_summary_path"
      fullspan_state_metric_set "promotion_eligible_count" "$(( $(fullspan_state_metric_get promotion_eligible_count 0) + 1 ))"
      if (( state_confirm_count < FULLSPAN_CONFIRM_MIN_REPLIES )); then
        log_decision_note "$queue_rel" "FULLSPAN_STRICT_PROMOTE_ELIGIBLE" "auto_confirm_count=$live_confirm_count" "vps_confirm_replays_complete"
      fi
    fi
    return 0
  fi

  if (( live_confirm_count != state_confirm_count )); then
    fullspan_state_set "$queue_rel" "$state_verdict" \
      "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
      "$rejection_reason" "$live_confirm_count" "$strict_gate_status" "$strict_gate_reason" "$run_groups_csv" "$strict_summary_path"
    if (( live_confirm_count > state_confirm_count )); then
      log_decision_note "$queue_rel" "FULLSPAN_STRICT_PENDING_CONFIRM" "confirm_count_update=$live_confirm_count" "waiting_for_vps_confirm_replay"
    fi
  fi

  return 0
}


low_yield_hardfail_stop_rule() {
  local queue_rel="$1"
  local strict_pass_count="$2"
  local strict_run_group_count="$3"
  local strict_rejection_reason="$4"
  local top_run_group="$5"
  local top_variant="$6"
  local top_score="$7"
  local strict_gate_reason="$8"
  local run_groups_csv="$9"
  local strict_summary_path="${10}"
  local confirm_count="${11}"

  if [[ -z "$queue_rel" ]]; then
    return 0
  fi

  local limit="${LOW_YIELD_HARDFAIL_STREAK_LIMIT}"
  [[ -n "$limit" ]] || limit=3
  local normalized_reason=""
  local code=""

  if [[ -n "$strict_rejection_reason" ]]; then
    normalized_reason="$strict_rejection_reason"
    code="$normalized_reason"
  else
    code="UNKNOWN"
  fi

  # Fail-closed only on repeated hard-fail signal with no evidence of improvement.
  if (( strict_pass_count > 0 )); then
    fullspan_state_queue_set "$queue_rel" \
      "hard_fail_streak" "0" \
      "hard_fail_last_reason" "" \
      "low_yield_fail_closed" "0" \
      "low_yield_fail_closed_reason" "" \
      "hard_fail_last_pass_count" "0" \
      "hard_fail_last_run_group_count" "0"
    return 0
  fi

  local prev_streak="$((0))"
  local prev_reason=""
  local prev_pass="0"
  local prev_groups="0"
  prev_streak="$(fullspan_state_get "$queue_rel" "hard_fail_streak" "0")"
  prev_reason="$(fullspan_state_get "$queue_rel" "hard_fail_last_reason" "")"
  prev_pass="$(fullspan_state_get "$queue_rel" "hard_fail_last_pass_count" "0")"
  prev_groups="$(fullspan_state_get "$queue_rel" "hard_fail_last_run_group_count" "0")"

  local next_streak=1
  if [[ -n "$prev_reason" && "$prev_reason" == "$code" ]]; then
    if awk -v a="$strict_pass_count" -v b="$prev_pass" -v c="$strict_run_group_count" -v d="$prev_groups" 'BEGIN { exit ((a<=b && c<=d)?0:1) }'; then
      next_streak=$((prev_streak + 1))
    fi
  fi

  if (( next_streak < 1 )); then
    next_streak=1
  fi

  fullspan_state_queue_set "$queue_rel" \
    "hard_fail_streak" "$next_streak" \
    "hard_fail_last_reason" "$code" \
    "hard_fail_last_pass_count" "$strict_pass_count" \
    "hard_fail_last_run_group_count" "$strict_run_group_count" \
    "low_yield_fail_closed" "0"

  fullspan_state_metric_inc "low_yield_hardfail_streak_eval" 1

  if (( next_streak >= limit )); then
    local stop_reason="LOW_YIELD_HARDFAIL_STREAK_${next_streak}::${code}"
    fullspan_state_set "$queue_rel" "REJECT" \
      "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
      "${strict_rejection_reason:-LOW_YIELD_HARDFAIL}" "$confirm_count" "FULLSPAN_PREFILTER_REJECT" "$strict_gate_reason" "$run_groups_csv" "$strict_summary_path"
    fullspan_state_queue_set "$queue_rel" \
      "low_yield_fail_closed" "1" \
      "low_yield_fail_closed_reason" "$stop_reason" \
      "hard_fail_streak" "$next_streak"
    fullspan_state_metric_inc "low_yield_hardfail_streak_stop" 1
    log_decision_note "$queue_rel" "LOW_YIELD_HARDFAIL_STOP" "reason=$stop_reason" "skip_and_fail_closed"
    return 1
  fi

  return 0
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
    fullspan_state_metric_inc "fullspan_cycle_cache_hit_count" 1
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

      low_yield_hardfail_stop_rule "$queue_rel" "$strict_pass_count" "$strict_run_group_count" "$strict_rejection_reason" "$top_run_group" "$top_variant" "$top_score" "" "$strict_run_groups" "$cycle_summary" "$confirm_count" || true
      write_decision_memo "$queue_rel" "$(fullspan_state_get "$queue_rel" "promotion_verdict" "ANALYZE")" "$strict_pass_count" "$strict_run_group_count" "$confirm_count" "$strict_rejection_reason" "$top_variant" "$top_run_group" "$top_score" "$(fullspan_state_get "$queue_rel" "strict_gate_status" "")" "$(fullspan_state_get "$queue_rel" "strict_gate_reason" "")"
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

    low_yield_hardfail_stop_rule "$queue_rel" "$strict_pass_count" "$strict_run_group_count" "$strict_rejection_reason" "$top_run_group" "$top_variant" "$top_score" "" "$strict_run_groups" "$cycle_summary" "$confirm_count" || true
    write_decision_memo "$queue_rel" "$(fullspan_state_get "$queue_rel" "promotion_verdict" "ANALYZE")" "$strict_pass_count" "$strict_run_group_count" "$confirm_count" "$strict_rejection_reason" "$top_variant" "$top_run_group" "$top_score" "$(fullspan_state_get "$queue_rel" "strict_gate_status" "")" "$(fullspan_state_get "$queue_rel" "strict_gate_reason" "")"
    fullspan_cycle_cache_set "$queue_rel" "$decision_fingerprint" "$strict_pass_count" "$strict_run_group_count" "$cycle_summary"
    printf '%s\n' "$queue_rel" >> "$FULLSPAN_CYCLE_STATE_FILE"
  else
    log "decision_cycle_fail queue=$queue_rel rc=$cycle_rc"
    fullspan_state_metric_set "decision_cycle_fail_count" "$(( $(fullspan_state_metric_get decision_cycle_fail_count 0) + 1 ))"
    write_decision_memo "$queue_rel" "REJECT" "$strict_pass_count" "$strict_run_group_count" "$confirm_count" "decision_cycle_fail_rc_${cycle_rc}" "$top_variant" "$top_run_group" "$top_score" "FULLSPAN_PREFILTER_UNKNOWN" ""
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
  hb_rate_per_min="0"

  out="$(python3 "$ROOT_DIR/scripts/optimization/_autonomous_heartbeat.py"     --state "$HEARTBEAT_STATE"     --queue "$queue_rel"     --pending "$pending"     --completed "$completed"     --total "$total"     --planned "$planned"     --running "$running"     --stalled "$stalled" 2>/dev/null || true)"

  if [[ -z "$out" ]]; then
    return
  fi

  IFS='|' read -r hb_queue hb_pending hb_completed hb_rate hb_eta hb_stale hb_done <<< "$out"
  hb_stale_sec="$hb_stale"
  hb_rate_per_min="$hb_rate"
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

quarantine_deterministic_stalled() {
  local queue_rel="$1"
  local queue_path="$ROOT_DIR/$queue_rel"
  python3 - "$queue_path" "$ROOT_DIR" "$DETERMINISTIC_QUARANTINE_FILE" <<'PY'
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

queue_path = Path(sys.argv[1])
root_dir = Path(sys.argv[2])
quarantine_file = Path(sys.argv[3])

payload = {"changed": 0, "codes": {}, "queue": str(queue_path)}
if not queue_path.exists():
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(0)

try:
    with queue_path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
except Exception:
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(0)

code_counter = Counter()
new_entries = []

for row in rows:
    status = str(row.get('status') or '').strip().lower()
    if status != 'stalled':
        continue

    results_dir = str(row.get('results_dir') or '').strip()
    if not results_dir:
        continue

    run_dir = Path(results_dir)
    if not run_dir.is_absolute():
        run_dir = (root_dir / run_dir).resolve()

    run_status = run_dir / 'run_status.json'
    run_log = run_dir / 'run.log'

    text_parts = []
    if run_status.exists():
        try:
            text_parts.append(run_status.read_text(encoding='utf-8', errors='ignore'))
        except Exception:
            pass
    if run_log.exists():
        try:
            text_parts.append(run_log.read_text(encoding='utf-8', errors='ignore'))
        except Exception:
            pass

    if not text_parts:
        continue

    blob = "\n".join(text_parts)
    code = None

    if ('backtest.max_var_multiplier' in blob and 'Input should be greater than 1' in blob):
        code = 'MAX_VAR_MULTIPLIER_INVALID'
    elif ('pydantic_core._pydantic_core.ValidationError' in blob and 'validation error for AppConfig' in blob):
        code = 'CONFIG_VALIDATION_ERROR'
    elif ('yaml' in blob.lower() and 'parse' in blob.lower() and 'error' in blob.lower()):
        code = 'CONFIG_PARSE_ERROR'

    if code is None:
        continue

    row['status'] = 'skipped'
    payload['changed'] += 1
    code_counter[code] += 1
    new_entries.append({
        'ts': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'queue': str(queue_path),
        'results_dir': str(run_dir),
        'run_id': str(Path(results_dir).name),
        'code': code,
    })

if payload['changed'] > 0 and rows:
    with queue_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    state = {'entries': []}
    if quarantine_file.exists():
        try:
            state = json.loads(quarantine_file.read_text(encoding='utf-8'))
        except Exception:
            state = {'entries': []}
    if not isinstance(state, dict):
        state = {'entries': []}
    entries = state.get('entries', [])
    if not isinstance(entries, list):
        entries = []

    seen = {(e.get('queue'), e.get('results_dir'), e.get('code')) for e in entries if isinstance(e, dict)}
    for e in new_entries:
        key = (e.get('queue'), e.get('results_dir'), e.get('code'))
        if key in seen:
            continue
        entries.append(e)
        seen.add(key)

    state['entries'] = entries[-20000:]
    quarantine_file.parent.mkdir(parents=True, exist_ok=True)
    quarantine_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')

payload['codes'] = dict(code_counter)
print(json.dumps(payload, ensure_ascii=False))
PY
}

prefilter_planned_gate_aware() {
  local queue_rel="$1"
  local queue_path="$ROOT_DIR/$queue_rel"
  python3 - "$queue_path" "$ROOT_DIR" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

queue_path = Path(sys.argv[1])
root_dir = Path(sys.argv[2])

payload = {"changed": 0, "codes": {}}
if not queue_path.exists():
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(0)

try:
    with queue_path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
except Exception:
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(0)

codes = Counter()
for row in rows:
    status = str(row.get('status') or '').strip().lower()
    if status != 'planned':
        continue

    cfg = str(row.get('config_path') or '').strip()
    if not cfg:
        continue
    cfg_path = Path(cfg)
    if not cfg_path.is_absolute():
        cfg_path = (root_dir / cfg_path).resolve()

    if not cfg_path.exists():
        row['status'] = 'skipped'
        payload['changed'] += 1
        codes['CONFIG_MISSING'] += 1
        continue

    try:
        data = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    except Exception:
        row['status'] = 'skipped'
        payload['changed'] += 1
        codes['CONFIG_PARSE_ERROR'] += 1
        continue

    if not isinstance(data, dict):
        row['status'] = 'skipped'
        payload['changed'] += 1
        codes['CONFIG_INVALID'] += 1
        continue

    bt = data.get('backtest') or {}
    mvm = bt.get('max_var_multiplier', None)
    try:
        mvm_f = float(mvm)
    except Exception:
        mvm_f = None

    if mvm_f is not None and mvm_f <= 1.0:
        row['status'] = 'skipped'
        payload['changed'] += 1
        codes['MAX_VAR_MULTIPLIER_INVALID'] += 1

if payload['changed'] > 0 and rows:
    with queue_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

payload['codes'] = dict(codes)
print(json.dumps(payload, ensure_ascii=False))
PY
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

  local reason_streak
  reason_streak="$(repair_reason_streak_record "$queue_rel" "${cause:-UNKNOWN}")"
  if (( reason_streak > SAME_REASON_REPAIR_CAP )); then
    local state_strict_pass_count state_strict_run_groups state_confirm_count
    state_strict_pass_count="$(fullspan_state_get "$queue_rel" "strict_pass_count" "0")"
    state_strict_run_groups="$(fullspan_state_get "$queue_rel" "strict_run_group_count" "0")"
    state_confirm_count="$(fullspan_state_get "$queue_rel" "confirm_count" "0")"
    fullspan_state_set "$queue_rel" "REJECT" "$state_strict_pass_count" "$state_strict_run_groups" "" "" "" "same_reason_repair_cap:${cause}" "$state_confirm_count" "FULLSPAN_PREFILTER_REJECT"
    fullspan_state_metric_inc "same_reason_repair_cap_hit" 1
    mark_orphan "$queue_rel" "same_reason_repair_cap_${cause}_${reason_streak}"
    log_decision_note "$queue_rel" "FULLSPAN_REPAIR_CAP_FAIL_CLOSED" "cause=${cause} streak=${reason_streak}" "skip_repeated_low_yield_repairs"
    log "repair_stalled_same_reason_cap queue=$queue_rel cause=$cause streak=$reason_streak cap=$SAME_REASON_REPAIR_CAP"
    return 1
  fi

  local quarantine_payload
  local quarantine_changed
  local quarantine_codes
  quarantine_payload="$(quarantine_deterministic_stalled "$queue_rel")"
  quarantine_changed="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(int(d.get("changed",0)))' "$quarantine_payload" 2>/dev/null || echo 0)"
  quarantine_codes="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); c=d.get("codes",{}); print(";".join(f"{k}:{v}" for k,v in sorted(c.items())))' "$quarantine_payload" 2>/dev/null || true)"
  if [[ -n "$quarantine_changed" && "$quarantine_changed" -gt 0 ]]; then
    log "deterministic_quarantine queue=$queue_rel changed=$quarantine_changed codes=${quarantine_codes:-none}"
    log_decision_note "$queue_rel" "FULLSPAN_DETERMINISTIC_QUARANTINE" "count=$quarantine_changed codes=${quarantine_codes:-none}" "skip_invalid_config_reruns"
    fullspan_state_metric_inc "deterministic_quarantine_count" "$quarantine_changed"
  fi

  local rc=0
  local max_retries
  max_retries="$(choose_max_retries "$cause")"

  local stalled_after_quarantine
  stalled_after_quarantine="$(python3 - "$ROOT_DIR/$queue_rel" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit(0)
try:
    rows = list(csv.DictReader(path.open(newline='', encoding='utf-8')))
except Exception:
    print(0)
    raise SystemExit(0)
stalled = 0
for row in rows:
    if str(row.get('status') or '').strip().lower() == 'stalled':
        stalled += 1
print(stalled)
PY
)"
  if [[ "$stalled_after_quarantine" == "0" ]]; then
    log "repair_stalled_noop queue=$queue_rel reason=deterministic_quarantine_exhausted"
    return 0
  fi

  if [[ -n "$SERVER_IP" ]]; then
    ensure_vps_ready "repair_stalled:$queue_rel" || true
    ("$ROOT_DIR/scripts/optimization/recover_stalled_queue.sh" --queue "$queue_rel" --parallel "$parallel" --compute-host "$SERVER_IP" --ssh-user "$SERVER_USER" --postprocess true --wait-completion false --max-retries "$max_retries") >>"$LOG_FILE" 2>&1 || rc=$?
  else
    ("$ROOT_DIR/scripts/optimization/recover_stalled_queue.sh" --queue "$queue_rel" --parallel "$parallel") >>"$LOG_FILE" 2>&1 || rc=$?
  fi

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

repair_reason_streak_record() {
  local queue_rel="$1"
  local reason="$2"
  local prev_reason="${stalled_repair_last_reason_by_queue[$queue_rel]:-}"
  local cur="${stalled_repair_same_reason_streak_by_queue[$queue_rel]:-0}"

  if [[ -n "$reason" && "$reason" == "$prev_reason" ]]; then
    cur=$((cur + 1))
  else
    cur=1
  fi

  stalled_repair_last_reason_by_queue["$queue_rel"]="$reason"
  stalled_repair_same_reason_streak_by_queue["$queue_rel"]="$cur"
  echo "$cur"
}

repair_reason_streak_reset() {
  local queue_rel="$1"
  stalled_repair_last_reason_by_queue["$queue_rel"]=""
  stalled_repair_same_reason_streak_by_queue["$queue_rel"]=0
}

repair_stalled_state_reset() {
  local queue_rel="$1"
  stalled_repair_failures["$queue_rel"]=0
  repair_reason_streak_reset "$queue_rel"
}

remote_runner_count() {
  local remote_count
  remote_count="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=6 "$SERVER_USER@$SERVER_IP" "pgrep -f 'watch_wfa_queue.sh|run_wfa_queue.py|python.*walk_forward' | awk -v self=\"\$\$\" '\$1 != self' | wc -l" || true)"
  if [[ -z "$remote_count" ]]; then
    echo 0
    return
  fi
  echo "$remote_count"
}

sla_watch_queue() {
  local queue_rel="$1"
  local pending="$2"
  local running="$3"
  local state_verdict="$4"

  if (( pending > 0 && running == 0 )); then
    local rc
    rc="$(remote_runner_count)"
    if [[ -z "$rc" || ! "$rc" =~ ^[0-9]+$ ]]; then
      rc=0
    fi
    if (( rc == 0 )); then
      local idle_streak="${vps_idle_pending_streak_by_queue[$queue_rel]:-0}"
      idle_streak=$((idle_streak + 1))
      vps_idle_pending_streak_by_queue["$queue_rel"]="$idle_streak"
      if (( idle_streak >= SLA_VPS_IDLE_PENDING_CYCLES )); then
        fullspan_state_metric_inc "sla_vps_idle_with_pending_trigger" 1
        log_decision_note "$queue_rel" "SLA_VPS_IDLE_WITH_PENDING" "streak=$idle_streak pending=$pending" "watchdog_or_manual_intervention"
      fi
    else
      vps_idle_pending_streak_by_queue["$queue_rel"]=0
    fi
  else
    vps_idle_pending_streak_by_queue["$queue_rel"]=0
  fi

  if [[ "$state_verdict" == "PROMOTE_PENDING_CONFIRM" || "$state_verdict" == "PROMOTE_DEFER_CONFIRM" ]]; then
    local last_decision_epoch now_epoch age_sec
    last_decision_epoch="$(fullspan_state_metric_get last_decision_epoch 0)"
    now_epoch="$(date +%s)"
    age_sec=$((now_epoch - last_decision_epoch))
    if (( last_decision_epoch > 0 && age_sec >= SLA_CONFIRM_PENDING_SEC )); then
      fullspan_state_metric_inc "sla_confirm_pending_overdue_trigger" 1
      log_decision_note "$queue_rel" "SLA_CONFIRM_PENDING_OVERDUE" "age_sec=$age_sec" "prioritize_confirm_replay"
    fi
  fi
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

fullspan_rollup_sync() {
  local queue_rel="$1"
  local reason="$2"
  local now_epoch
  local last_epoch=0
  now_epoch="$(date +%s)"

  if [[ -f "$FULLSPAN_ROLLUP_SYNC_MARKER" ]]; then
    last_epoch="$(awk 'NR==1 {print $1}' "$FULLSPAN_ROLLUP_SYNC_MARKER" 2>/dev/null || true)"
    if [[ -z "$last_epoch" ]]; then
      last_epoch=0
    fi
  fi

  if (( now_epoch - last_epoch < FULLSPAN_ROLLUP_SYNC_MIN_INTERVAL )); then
    return 0
  fi

  sync_queue_status "$queue_rel"
  (cd "$ROOT_DIR" && ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup --no-auto-sync-status) >>"$LOG_FILE" 2>&1 || true
  printf '%s %s %s\n' "$now_epoch" "$queue_rel" "${reason:-milestone}" > "$FULLSPAN_ROLLUP_SYNC_MARKER"
  log "fullspan_rollup_sync queue=$queue_rel reason=${reason:-milestone}"
}

vps_is_reachable() {
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=6 "$SERVER_USER@$SERVER_IP" 'echo ok' >/dev/null 2>&1
}

ensure_vps_ready() {
  local reason="${1:-auto}"
  local now_epoch
  now_epoch="$(date +%s)"
  local last_recover
  last_recover="$(fullspan_state_metric_get vps_recover_last_epoch 0)"
  if [[ -z "$last_recover" ]]; then
    last_recover=0
  fi

  if vps_is_reachable; then
    return 0
  fi

  if (( now_epoch - last_recover < 300 )); then
    return 1
  fi

  fullspan_state_metric_set "vps_recover_last_epoch" "$now_epoch"
  log "vps_recover_attempt reason=$reason"
  if timeout 180 env SKIP_POWER=0 STOP_AFTER=0 UPDATE_CODE=0 SYNC_BACK=0 SYNC_UP=0 "$ROOT_DIR/scripts/remote/run_server_job.sh" echo ping >>"$LOG_FILE" 2>&1; then
    fullspan_state_metric_inc "vps_recover_success_count" 1
    log "vps_recover_success reason=$reason"
    return 0
  fi

  fullspan_state_metric_inc "vps_recover_fail_count" 1
  log_decision_note "global" "VPS_RECOVER_FAIL" "reason=$reason" "await_next_watchdog_cycle"
  log "vps_recover_fail reason=$reason"
  return 1
}

fullspan_confirm_fastlane_pending() {
  local queue_rel="$1"
  local qpath="$ROOT_DIR/$queue_rel"
  python3 - "$qpath" <<'PY'
import csv
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    print(0)
    raise SystemExit(0)
try:
    rows = list(csv.DictReader(p.open(newline='', encoding='utf-8')))
except Exception:
    print(0)
    raise SystemExit(0)
pending = 0
for r in rows:
    s = str(r.get('status') or '').strip().lower()
    if s in {'planned', 'running', 'stalled', 'failed', 'error'}:
        pending += 1
print(pending)
PY
}

trigger_confirm_fastlane() {
  local source_queue_rel="$1"
  local queue_name="$2"
  local top_variant="$3"
  local strict_pass_count="$4"
  local strict_run_groups="$5"
  local confirm_count="$6"

  if [[ -z "$top_variant" ]]; then
    return 0
  fi
  if (( strict_pass_count <= 0 || strict_run_groups < FULLSPAN_CONFIRM_MIN_GROUPS || confirm_count >= FULLSPAN_CONFIRM_MIN_REPLIES )); then
    return 0
  fi

  local now_epoch last_trigger
  now_epoch="$(date +%s)"
  last_trigger="$(fullspan_state_get "$source_queue_rel" "confirm_fastlane_last_trigger_epoch" "0")"
  if [[ -z "$last_trigger" ]]; then
    last_trigger=0
  fi
  if (( now_epoch - last_trigger < CONFIRM_FASTLANE_COOLDOWN_SEC )); then
    return 0
  fi

  local existing_confirm_queue
  existing_confirm_queue="$(fullspan_state_get "$source_queue_rel" "confirm_fastlane_queue_rel" "")"
  if [[ -n "$existing_confirm_queue" ]]; then
    local existing_pending
    existing_pending="$(fullspan_confirm_fastlane_pending "$existing_confirm_queue")"
    if (( existing_pending > 0 )); then
      return 0
    fi
  fi

  local safe_name
  safe_name="$(printf '%s' "$queue_name" | tr '/.' '__')"
  local shortlist_rel="artifacts/wfa/aggregate/.autonomous/confirm_fastlane_shortlist_${safe_name}.csv"
  local confirm_queue_dir_rel="artifacts/wfa/aggregate/confirm_fastlane_${safe_name}"
  local confirm_queue_rel="${confirm_queue_dir_rel}/run_queue.csv"
  local stress_dir_rel="configs/confirm_fastlane/${safe_name}/stress"
  local cycle_name="confirm_fastlane_${safe_name}"

  python3 - "$ROOT_DIR/$source_queue_rel" "$ROOT_DIR/$shortlist_rel" "$top_variant" <<'PY'
import csv
import sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])
needle = (sys.argv[3] or '').strip().lower()

rows = []
if src.exists():
    try:
        rows = list(csv.DictReader(src.open(newline='', encoding='utf-8')))
    except Exception:
        rows = []

picked = []
seen = set()
for r in rows:
    cfg = str(r.get('config_path') or '').strip()
    if not cfg or cfg in seen:
        continue
    if cfg.endswith('_stress.yaml'):
        continue
    blob = " ".join([cfg, str(r.get('results_dir') or ''), str(r.get('status') or '')]).lower()
    if needle and needle not in blob:
        continue
    seen.add(cfg)
    picked.append({'config_path': cfg, 'results_dir': str(r.get('results_dir') or ''), 'status': 'planned'})

if not picked:
    for r in rows:
        cfg = str(r.get('config_path') or '').strip()
        if not cfg or cfg in seen:
            continue
        if cfg.endswith('_stress.yaml'):
            continue
        seen.add(cfg)
        picked.append({'config_path': cfg, 'results_dir': str(r.get('results_dir') or ''), 'status': 'planned'})
        break

out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['config_path', 'results_dir', 'status'])
    w.writeheader()
    for row in picked[:1]:
        w.writerow(row)
print(len(picked[:1]))
PY

  local shortlist_count
  shortlist_count="$(python3 - "$ROOT_DIR/$shortlist_rel" <<'PY'
import csv
import sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
    print(0)
    raise SystemExit(0)
rows=list(csv.DictReader(p.open(newline='',encoding='utf-8')))
print(len(rows))
PY
)"
  if (( shortlist_count <= 0 )); then
    return 0
  fi

  (cd "$ROOT_DIR" && ./.venv/bin/python scripts/optimization/build_confirm_queue.py \
      --shortlist-queue "$shortlist_rel" \
      --cycle "$cycle_name" \
      --queue-dir "$confirm_queue_dir_rel" \
      --stress-config-dir "$stress_dir_rel" \
      --limit "$CONFIRM_FASTLANE_LIMIT") >>"$LOG_FILE" 2>&1 || return 0

  fullspan_state_queue_set "$source_queue_rel" \
    "confirm_fastlane_queue_rel" "$confirm_queue_rel" \
    "confirm_fastlane_last_trigger_epoch" "$now_epoch" \
    "confirm_pending_since_epoch" "${now_epoch}"

  local qlog="$STATE_DIR/confirm_fastlane_${safe_name}_$(date -u +%Y%m%d_%H%M%S).log"
  (
    cd "$ROOT_DIR"
    AUTONOMOUS_MODE=1 \
    ALLOW_HEAVY_RUN=1 \
    ./.venv/bin/python scripts/optimization/run_wfa_queue_powered.py \
      --queue "$confirm_queue_rel" \
      --compute-host "$SERVER_IP" \
      --ssh-user "$SERVER_USER" \
      --parallel "$CONFIRM_FASTLANE_PARALLEL" \
      --statuses planned \
      --max-retries 2 \
      --watchdog true \
      --wait-completion false \
      --postprocess true \
      --poweroff "$POWEROFF_AFTER_RUN" \
      >>"$qlog" 2>&1
  ) &

  fullspan_state_metric_inc "confirm_fastlane_trigger_count" 1
  log_decision_note "$source_queue_rel" "CONFIRM_FASTLANE_TRIGGER" "confirm_queue=$confirm_queue_rel variant=$top_variant" "await_confirm_replay"
  log "confirm_fastlane_trigger source=$source_queue_rel confirm_queue=$confirm_queue_rel variant=$top_variant"
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
  ensure_vps_ready "start_queue:$queue_rel" || true
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
      --poweroff "$POWEROFF_AFTER_RUN" \
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
declare -A stalled_repair_last_reason_by_queue
declare -A stalled_repair_same_reason_streak_by_queue
declare -A no_progress_streak_by_queue
declare -A vps_idle_pending_streak_by_queue

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
    if fallback_pending_candidate; then
      log "candidate_fallback_selected reason=no_candidate_file"
    else
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
  fi

  IFS=',' read -r queue planned running stalled failed completed total urgency mtime promotion_potential gate_status gate_reason pre_rank_score strict_gate_status strict_gate_reason < <(tail -n 1 "$CANDIDATE_FILE")

  if [[ -z "${queue:-}" || "$queue" == "queue" ]]; then
    if fallback_pending_candidate; then
      log "candidate_parse_fallback reason=parse_empty"
      IFS=',' read -r queue planned running stalled failed completed total urgency mtime promotion_potential gate_status gate_reason pre_rank_score strict_gate_status strict_gate_reason < <(tail -n 1 "$CANDIDATE_FILE")
    fi
  fi

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

  if (( planned > 0 )); then
    gate_prefilter_payload="$(prefilter_planned_gate_aware "$queue_rel")"
    gate_prefilter_changed="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(int(d.get("changed",0)))' "$gate_prefilter_payload" 2>/dev/null || echo 0)"
    gate_prefilter_codes="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); c=d.get("codes",{}); print(";".join(f"{k}:{v}" for k,v in sorted(c.items())))' "$gate_prefilter_payload" 2>/dev/null || true)"
    if [[ -n "$gate_prefilter_changed" && "$gate_prefilter_changed" -gt 0 ]]; then
      fullspan_state_metric_inc "gate_aware_generation_skip_count" "$gate_prefilter_changed"
      log_decision_note "$queue_rel" "GATE_AWARE_GENERATION_SKIP" "count=$gate_prefilter_changed codes=${gate_prefilter_codes:-none}" "prefilter_planned_invalid_configs"
      log "gate_aware_prefilter queue=$queue_rel changed=$gate_prefilter_changed codes=${gate_prefilter_codes:-none}"
      read -r planned running stalled failed completed total <<< "$(python3 - "$ROOT_DIR/$queue_rel" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path
p=Path(sys.argv[1])
rows=list(csv.DictReader(p.open(newline='',encoding='utf-8'))) if p.exists() else []
c=Counter((r.get('status') or '').strip().lower() for r in rows)
print(c.get('planned',0), c.get('running',0), c.get('stalled',0), c.get('failed',0)+c.get('error',0), c.get('completed',0), len(rows))
PY
)"
      pending=$((planned + running + stalled + failed))
    fi
  fi

  promotion_verdict="ANALYZE"
  state_verdict="$(fullspan_state_get "$queue_rel" "promotion_verdict" "ANALYZE")"
  state_rejection_reason="$(fullspan_state_get "$queue_rel" "rejection_reason" "")"
  state_strict_pass_count="$(fullspan_state_get "$queue_rel" "strict_pass_count" "0")"
  state_strict_gate_status="$(fullspan_state_get "$queue_rel" "strict_gate_status" "FULLSPAN_PREFILTER_UNKNOWN")"
  state_strict_gate_reason="$(fullspan_state_get "$queue_rel" "strict_gate_reason" "")"
  state_strict_run_groups="$(fullspan_state_get "$queue_rel" "strict_run_group_count" "0")"
  state_confirm_count="$(fullspan_state_get "$queue_rel" "confirm_count" "0")"
  state_top_variant="$(fullspan_state_get "$queue_rel" "top_variant" "")"
  low_yield_fail_closed="$(fullspan_state_get "$queue_rel" "low_yield_fail_closed" "0")"
  low_yield_fail_closed_reason="$(fullspan_state_get "$queue_rel" "low_yield_fail_closed_reason" "")"

  fullspan_reconcile_confirm_progress "$queue_rel"

  state_verdict="$(fullspan_state_get "$queue_rel" "promotion_verdict" "ANALYZE")"
  state_rejection_reason="$(fullspan_state_get "$queue_rel" "rejection_reason" "")"
  state_strict_pass_count="$(fullspan_state_get "$queue_rel" "strict_pass_count" "0")"
  state_strict_gate_status="$(fullspan_state_get "$queue_rel" "strict_gate_status" "FULLSPAN_PREFILTER_UNKNOWN")"
  state_strict_gate_reason="$(fullspan_state_get "$queue_rel" "strict_gate_reason" "")"
  state_strict_run_groups="$(fullspan_state_get "$queue_rel" "strict_run_group_count" "0")"
  state_confirm_count="$(fullspan_state_get "$queue_rel" "confirm_count" "0")"
  state_top_variant="$(fullspan_state_get "$queue_rel" "top_variant" "")"

  # If reject was caused by no-progress while queue is planned-only, reopen for execution.
  if [[ "$state_verdict" == "REJECT" && "$state_rejection_reason" == "no_progress_streak" && "$planned" -gt 0 && "$running" -eq 0 && "$stalled" -eq 0 ]]; then
    fullspan_state_queue_set "$queue_rel" "promotion_verdict" "ANALYZE" "rejection_reason" ""
    state_verdict="ANALYZE"
    state_rejection_reason=""
    state_strict_gate_status="FULLSPAN_PREFILTER_UNKNOWN"
    log "reopen_from_no_progress queue=$queue_rel planned=$planned"
  fi

  if [[ "$state_verdict" == "PROMOTE_PENDING_CONFIRM" || "$state_verdict" == "PROMOTE_DEFER_CONFIRM" ]]; then
    pending_since_epoch="$(fullspan_state_get "$queue_rel" "confirm_pending_since_epoch" "0")"
    if [[ -z "$pending_since_epoch" || "$pending_since_epoch" == "0" ]]; then
      fullspan_state_queue_set "$queue_rel" "confirm_pending_since_epoch" "$(date +%s)"
    fi
  else
    fullspan_state_queue_set "$queue_rel" "confirm_pending_since_epoch" "0"
  fi

  trigger_confirm_fastlane "$queue_rel" "$(basename "$(dirname "$queue_rel")")" "$state_top_variant" "$state_strict_pass_count" "$state_strict_run_groups" "$state_confirm_count"

  if [[ "$low_yield_fail_closed" == "1" ]]; then
    promotion_verdict="REJECT"
    promotion_potential="REJECT"
    gate_status="HARD_FAIL"
    gate_reason="${low_yield_fail_closed_reason:-LOW_YIELD_HARDFAIL_STREAK}"
  fi

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

  if [[ "$state_verdict" == "PROMOTE_PENDING_CONFIRM" || "$state_verdict" == "PROMOTE_DEFER_CONFIRM" ]]; then
    if (( state_strict_pass_count > 0 && state_strict_run_groups >= FULLSPAN_CONFIRM_MIN_GROUPS )); then
      fullspan_rollup_sync "$queue_rel" "confirm_fastlane_watch"
    fi
  fi

  if [[ -n "$prev_queue" && "$prev_queue" == "$queue_rel" ]]; then
    if [[ "$pending" -lt "$prev_pending" ]]; then
      clear_orphan "$queue_rel"
      no_progress_streak_by_queue["$queue_rel"]=0
      log "progress_seen queue=$queue_rel prev=$prev_pending curr=$pending"
      fullspan_rollup_sync "$queue_rel" "progress_milestone"
    fi
  fi

  heartbeat_update "$queue_rel" "$pending" "$completed" "$total" "$planned" "$running" "$stalled"
  sla_watch_queue "$queue_rel" "$pending" "$running" "$state_verdict"

  if (( pending > 0 )); then
    if (( running == 0 )) && ! vps_is_reachable; then
      no_progress_streak_by_queue["$queue_rel"]=0
      log "no_progress_pause queue=$queue_rel reason=vps_unreachable pending=$pending"
      sleep 5
      continue
    fi

    if (( running > 0 || stalled > 0 )); then
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
  else
    no_progress_streak_by_queue["$queue_rel"]=0
  fi

  safe_queue_name="$(basename "$queue")"
  if [[ "$pending" -eq 0 && "$completed" -gt 0 && "$promotion_verdict" != "REJECT" ]]; then
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
      if repair_stalled_queue "$queue_rel" "$planned" "$running" "$stalled" "$total" "$reason"; then
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
