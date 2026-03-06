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
VPS_BATCH_SESSION_ENABLE="${VPS_BATCH_SESSION_ENABLE:-1}"
VPS_BATCH_SESSION_MAX_SECONDS="${VPS_BATCH_SESSION_MAX_SECONDS:-3600}"
VPS_BATCH_SESSION_MAX_JOBS="${VPS_BATCH_SESSION_MAX_JOBS:-6}"
VPS_BATCH_SESSION_IDLE_GRACE_SEC="${VPS_BATCH_SESSION_IDLE_GRACE_SEC:-180}"
VPS_BATCH_SESSION_STOP_COOLDOWN_SEC="${VPS_BATCH_SESSION_STOP_COOLDOWN_SEC:-180}"
LOCK_FILE="$STATE_DIR/driver.lock"
CANDIDATE_FILE="$STATE_DIR/candidate.csv"
READY_BUFFER_POOL_FILE="$STATE_DIR/candidate_pool.csv"
READY_BUFFER_STATE_FILE="$STATE_DIR/ready_queue_buffer.json"
COLD_FAIL_STATE_FILE="$STATE_DIR/cold_fail_index.json"
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
RANKING_PRIMARY_KEY="${RANKING_PRIMARY_KEY:-score_fullspan_v1}"
DECISION_NOTES_FILE="$STATE_DIR/decision_notes.jsonl"
DECISION_MEMO_DIR="$STATE_DIR/decision_memos"
DETERMINISTIC_QUARANTINE_FILE="$STATE_DIR/deterministic_quarantine.json"
GATE_SURROGATE_STATE_FILE="$STATE_DIR/gate_surrogate_state.json"
SEARCH_DIRECTOR_DIRECTIVE_FILE="$STATE_DIR/search_director_directive.json"
YIELD_GOVERNOR_STATE_FILE="$STATE_DIR/yield_governor_state.json"


FULLSPAN_CYCLE_STATE_FILE="$STATE_DIR/mini_cycle_state.txt"
FULLSPAN_CYCLE_CACHE_FILE="$STATE_DIR/fullspan_cycle_cache.json"
FULLSPAN_DECISION_STATE_FILE="$STATE_DIR/fullspan_decision_state.json"
VPS_RECOVERY_STATE_FILE="$STATE_DIR/vps_recovery_state.json"
CONFIRM_LINEAGE_REGISTRY_FILE="$STATE_DIR/confirm_lineage_registry.json"
FULLSPAN_CONFIRM_MIN_GROUPS="${FULLSPAN_CONFIRM_MIN_GROUPS:-2}"
FULLSPAN_CONFIRM_MIN_REPLIES="${FULLSPAN_CONFIRM_MIN_REPLIES:-2}"
FULLSPAN_ROLLUP_SYNC_MIN_INTERVAL="${FULLSPAN_ROLLUP_SYNC_MIN_INTERVAL:-300}"
FULLSPAN_ROLLUP_SYNC_MARKER="$STATE_DIR/fullspan_rollup_sync.marker"
CAPACITY_CONTROLLER_STATE_FILE="$STATE_DIR/capacity_controller_state.json"
CONFIRM_SLA_ESCALATION_STATE_FILE="$STATE_DIR/confirm_sla_escalation_state.json"
BATCH_SESSION_STATE_FILE="$STATE_DIR/batch_session_state.json"
DRIVER_CONFIRM_FASTLANE_ENABLE="${DRIVER_CONFIRM_FASTLANE_ENABLE:-1}"
VPS_HOT_STANDBY_ENABLE="${VPS_HOT_STANDBY_ENABLE:-1}"
VPS_HOT_STANDBY_TTL_SEC="${VPS_HOT_STANDBY_TTL_SEC:-2700}"
NO_PROGRESS_STALE_CYCLES="${NO_PROGRESS_STALE_CYCLES:-6}"
SAME_REASON_REPAIR_CAP="${SAME_REASON_REPAIR_CAP:-3}"
ADAPTIVE_LOW_RATE_THRESHOLD="${ADAPTIVE_LOW_RATE_THRESHOLD:-0.05}"
ADAPTIVE_HIGH_RATE_THRESHOLD="${ADAPTIVE_HIGH_RATE_THRESHOLD:-0.60}"
SLA_VPS_IDLE_PENDING_CYCLES="${SLA_VPS_IDLE_PENDING_CYCLES:-3}"
SLA_CONFIRM_PENDING_SEC="${SLA_CONFIRM_PENDING_SEC:-7200}"
VPS_RECOVER_TIMEOUT_SEC="${VPS_RECOVER_TIMEOUT_SEC:-420}"
VPS_RECOVER_BASE_COOLDOWN_SEC="${VPS_RECOVER_BASE_COOLDOWN_SEC:-300}"
VPS_RECOVER_MAX_COOLDOWN_SEC="${VPS_RECOVER_MAX_COOLDOWN_SEC:-3600}"
VPS_RECOVER_FAIL_HARD_NOTE_STREAK="${VPS_RECOVER_FAIL_HARD_NOTE_STREAK:-4}"
VPS_RECOVER_ACTIVE_SSH_DOWN_FASTPATH="${VPS_RECOVER_ACTIVE_SSH_DOWN_FASTPATH:-1}"
VPS_FORCE_CYCLE_STREAK="${VPS_FORCE_CYCLE_STREAK:-3}"
VPS_FORCE_CYCLE_COOLDOWN_SEC="${VPS_FORCE_CYCLE_COOLDOWN_SEC:-1800}"
VPS_FORCE_CYCLE_SHUTDOWN_WAIT_SEC="${VPS_FORCE_CYCLE_SHUTDOWN_WAIT_SEC:-20}"
VPS_FORCE_CYCLE_BOOT_WAIT_SEC="${VPS_FORCE_CYCLE_BOOT_WAIT_SEC:-360}"
VPS_UNREACHABLE_ORPHAN_STREAK="${VPS_UNREACHABLE_ORPHAN_STREAK:-3}"
VPS_UNREACHABLE_SLEEP_CAP_SEC="${VPS_UNREACHABLE_SLEEP_CAP_SEC:-300}"
EARLY_STOP_MIN_COMPLETED="${EARLY_STOP_MIN_COMPLETED:-8}"
EARLY_STOP_FAIL_FRACTION="${EARLY_STOP_FAIL_FRACTION:-0.75}"
EARLY_STOP_DOMINANT_FRACTION="${EARLY_STOP_DOMINANT_FRACTION:-0.70}"
EARLY_STOP_DOMINANT_MIN="${EARLY_STOP_DOMINANT_MIN:-6}"
EARLY_ABORT_MIN_COMPLETED="${EARLY_ABORT_MIN_COMPLETED:-12}"
EARLY_ABORT_ZERO_ACTIVITY_SHARE="${EARLY_ABORT_ZERO_ACTIVITY_SHARE:-0.80}"
EARLY_ABORT_ZERO_ACTIVITY_MIN="${EARLY_ABORT_ZERO_ACTIVITY_MIN:-6}"
CONFIRM_FASTLANE_LIMIT="${CONFIRM_FASTLANE_LIMIT:-1}"
CONFIRM_FASTLANE_PARALLEL="${CONFIRM_FASTLANE_PARALLEL:-2}"
CONFIRM_FASTLANE_COOLDOWN_SEC="${CONFIRM_FASTLANE_COOLDOWN_SEC:-1800}"
SEARCH_PARALLEL_ABS_MAX="${SEARCH_PARALLEL_ABS_MAX:-48}"
LOW_YIELD_HARDFAIL_STREAK_LIMIT="${LOW_YIELD_HARDFAIL_STREAK_LIMIT:-3}"
LOW_YIELD_HOMOGENEOUS_FAIL_FRACTION="${LOW_YIELD_HOMOGENEOUS_FAIL_FRACTION:-0.70}"
LOW_YIELD_HOMOGENEOUS_FAIL_MIN="${LOW_YIELD_HOMOGENEOUS_FAIL_MIN:-2}"
SELECTOR_STALLED_BUDGET_RATIO="${SELECTOR_STALLED_BUDGET_RATIO:-0.20}"
MIN_PLANNED_BACKLOG="${MIN_PLANNED_BACKLOG:-12}"
AUTO_SEED_COOLDOWN_SEC="${AUTO_SEED_COOLDOWN_SEC:-900}"
AUTO_SEED_PENDING_THRESHOLD="${AUTO_SEED_PENDING_THRESHOLD:-96}"
AUTO_SEED_NUM_VARIANTS="${AUTO_SEED_NUM_VARIANTS:-64}"
AUTO_SEED_NUM_VARIANTS_FLOOR="${AUTO_SEED_NUM_VARIANTS_FLOOR:-24}"
READY_BUFFER_TARGET_DEPTH="${READY_BUFFER_TARGET_DEPTH:-3}"
READY_BUFFER_REFILL_THRESHOLD="${READY_BUFFER_REFILL_THRESHOLD:-2}"
READY_BUFFER_MAX_AGE_SEC="${READY_BUFFER_MAX_AGE_SEC:-900}"
READY_BUFFER_OVERLAP_TAIL_PENDING="${READY_BUFFER_OVERLAP_TAIL_PENDING:-24}"
READY_BUFFER_MAX_ACTIVE_REMOTE_QUEUES="${READY_BUFFER_MAX_ACTIVE_REMOTE_QUEUES:-2}"
HARD_FAIL_COLD_TTL_SEC="${HARD_FAIL_COLD_TTL_SEC:-21600}"
NO_PROGRESS_BREAKER_WINDOW_SEC="${NO_PROGRESS_BREAKER_WINDOW_SEC:-900}"
NO_PROGRESS_BREAKER_STREAK="${NO_PROGRESS_BREAKER_STREAK:-2}"
NO_PROGRESS_BREAKER_FRESH_QUEUE_GRACE_SEC="${NO_PROGRESS_BREAKER_FRESH_QUEUE_GRACE_SEC:-1800}"
FORCE_SYNC_BEFORE_START="${FORCE_SYNC_BEFORE_START:-1}"
LAST_REJECTED_QUEUE="${LAST_REJECTED_QUEUE:-}"
DRIVER_MAX_LOCAL_SEARCH_RUNNERS="${DRIVER_MAX_LOCAL_SEARCH_RUNNERS:-3}"
DRIVER_MAX_REMOTE_RUNNERS="${DRIVER_MAX_REMOTE_RUNNERS:-64}"
VPS_HOT_STANDBY_GRACE_SEC="${VPS_HOT_STANDBY_GRACE_SEC:-900}"
RUNTIME_OBSERVABILITY_WINDOW_SEC="${RUNTIME_OBSERVABILITY_WINDOW_SEC:-1800}"
REPLAY_FASTLANE_SCAN_LIMIT="${REPLAY_FASTLANE_SCAN_LIMIT:-2}"
RUNTIME_OBSERVABILITY_STATE_FILE="$STATE_DIR/runtime_observability_state.json"

mkdir -p "$STATE_DIR"

vps_runtime_fail_streak=0
vps_runtime_unreachable_since=0
vps_runtime_last_recover_epoch=0
vps_runtime_next_retry_epoch=0
vps_runtime_last_force_cycle_epoch=0

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
trap 'batch_session_save_state; flock -u 9; rm -f "$LOCK_FILE"' EXIT

batch_session_active=0
batch_session_start_epoch=0
batch_session_runs_started=0
batch_session_last_dispatch_epoch=0
batch_session_last_stop_epoch=0

is_truthy() {
  local raw
  raw="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  [[ "$raw" == "1" || "$raw" == "true" || "$raw" == "yes" || "$raw" == "y" || "$raw" == "on" ]]
}

batch_session_enabled() {
  is_truthy "$VPS_BATCH_SESSION_ENABLE"
}

batch_session_save_state() {
  python3 - "$BATCH_SESSION_STATE_FILE" "$batch_session_active" "$batch_session_start_epoch" "$batch_session_runs_started" "$batch_session_last_dispatch_epoch" "$batch_session_last_stop_epoch" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])

def to_int(value):
    try:
        return int(float(value or 0))
    except Exception:
        return 0

payload = {
    "active": bool(to_int(sys.argv[2])),
    "start_epoch": to_int(sys.argv[3]),
    "runs_started": to_int(sys.argv[4]),
    "last_dispatch_epoch": to_int(sys.argv[5]),
    "last_stop_epoch": to_int(sys.argv[6]),
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

batch_session_load_state() {
  if [[ ! -f "$BATCH_SESSION_STATE_FILE" ]]; then
    return 0
  fi
  local loaded
  loaded="$(python3 - "$BATCH_SESSION_STATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("0,0,0,0,0")
    raise SystemExit(0)
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("0,0,0,0,0")
    raise SystemExit(0)

def to_int(value):
    try:
        return int(float(value or 0))
    except Exception:
        return 0

print(
    "{},{},{},{},{}".format(
        1 if bool(data.get("active")) else 0,
        to_int(data.get("start_epoch")),
        to_int(data.get("runs_started")),
        to_int(data.get("last_dispatch_epoch")),
        to_int(data.get("last_stop_epoch")),
    )
)
PY
)"
  IFS=',' read -r batch_session_active batch_session_start_epoch batch_session_runs_started batch_session_last_dispatch_epoch batch_session_last_stop_epoch <<< "$loaded"
}

batch_session_note_dispatch() {
  if ! batch_session_enabled; then
    return 0
  fi
  local now_epoch
  now_epoch="$(date +%s)"
  if (( batch_session_active == 0 )); then
    batch_session_active=1
    batch_session_start_epoch="$now_epoch"
    batch_session_runs_started=0
    log "batch_session_start ts=$now_epoch"
  fi
  batch_session_runs_started=$((batch_session_runs_started + 1))
  batch_session_last_dispatch_epoch="$now_epoch"
  batch_session_save_state
}

batch_session_queue_poweroff() {
  if batch_session_enabled; then
    echo "false"
  else
    echo "$POWEROFF_AFTER_RUN"
  fi
}

global_pending_count() {
  python3 - "$QUEUE_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
pending = 0
for queue in root.rglob("run_queue.csv"):
    if "/rollup/" in str(queue) or "/.autonomous/" in str(queue):
        continue
    try:
        with queue.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        continue
    for row in rows:
        status = str(row.get("status") or "").strip().lower()
        if status in {"planned", "running", "stalled", "failed", "error"}:
            pending += 1
print(pending)
PY
}

batch_session_stop() {
  local reason="${1:-idle}"
  local now_epoch
  now_epoch="$(date +%s)"

  if (( batch_session_active == 0 )); then
    return 0
  fi
  if (( now_epoch - batch_session_last_stop_epoch < VPS_BATCH_SESSION_STOP_COOLDOWN_SEC )); then
    return 0
  fi
  if ! vps_is_reachable; then
    batch_session_active=0
    batch_session_start_epoch=0
    batch_session_runs_started=0
    batch_session_last_dispatch_epoch=0
    batch_session_last_stop_epoch="$now_epoch"
    batch_session_save_state
    return 0
  fi

  log "batch_session_stop_attempt reason=$reason"
  if timeout 180 env SKIP_POWER=1 STOP_AFTER=1 STOP_VIA_SSH=1 UPDATE_CODE=0 SYNC_BACK=0 SYNC_UP=0 ALLOW_STOP_DURING_ACTIVE_BATCH=1 "$ROOT_DIR/scripts/remote/run_server_job.sh" echo batch_session_stop >>"$LOG_FILE" 2>&1; then
    batch_session_last_stop_epoch="$now_epoch"
    log "batch_session_stop_success reason=$reason runs_started=$batch_session_runs_started"
  else
    batch_session_last_stop_epoch="$now_epoch"
    log "batch_session_stop_fail reason=$reason"
  fi

  batch_session_active=0
  batch_session_start_epoch=0
  batch_session_runs_started=0
  batch_session_last_dispatch_epoch=0
  batch_session_save_state
}

batch_session_maybe_stop() {
  local reason="${1:-idle}"
  if ! batch_session_enabled; then
    return 0
  fi
  if (( batch_session_active == 0 )); then
    return 0
  fi

  local now_epoch
  now_epoch="$(date +%s)"
  local remote_count cpu_busy_without_queue_job
  remote_count="$(remote_active_queue_jobs)"
  cpu_busy_without_queue_job="$(remote_cpu_busy_without_queue_job)"
  [[ "$remote_count" =~ ^[0-9]+$ ]] || remote_count=0
  [[ "$cpu_busy_without_queue_job" =~ ^[0-9]+$ ]] || cpu_busy_without_queue_job=0
  if (( remote_count > 0 || cpu_busy_without_queue_job == 1 )); then
    return 0
  fi

  local pending_total
  pending_total="$(global_pending_count)"
  if [[ -z "$pending_total" || ! "$pending_total" =~ ^[0-9]+$ ]]; then
    pending_total=0
  fi

  local ready_depth replay_pending standby_ttl
  ready_depth="$(ready_buffer_depth)"
  replay_pending="$(confirm_fastlane_pending_count)"
  standby_ttl="${VPS_HOT_STANDBY_GRACE_SEC:-${VPS_HOT_STANDBY_TTL_SEC:-2700}}"
  [[ "$ready_depth" =~ ^[0-9]+$ ]] || ready_depth=0
  [[ "$replay_pending" =~ ^[0-9]+$ ]] || replay_pending=0
  [[ "$standby_ttl" =~ ^[0-9]+$ ]] || standby_ttl=2700

  local age_sec=$((now_epoch - batch_session_start_epoch))
  local idle_sec=$((now_epoch - batch_session_last_dispatch_epoch))

  if (( pending_total == 0 )); then
    batch_session_stop "all_queues_done:${reason}"
    return 0
  fi

  if is_hot_standby_enabled; then
    if (( ready_depth > 0 || replay_pending > 0 || idle_sec < standby_ttl )); then
      log "batch_session_hot_standby reason=$reason pending_total=$pending_total ready_depth=$ready_depth replay_pending=$replay_pending idle_sec=$idle_sec standby_ttl=$standby_ttl"
      return 0
    fi
  fi

  if (( batch_session_runs_started >= VPS_BATCH_SESSION_MAX_JOBS && idle_sec >= VPS_BATCH_SESSION_IDLE_GRACE_SEC )); then
    batch_session_stop "max_jobs:${reason}"
    return 0
  fi

  if (( age_sec >= VPS_BATCH_SESSION_MAX_SECONDS && idle_sec >= VPS_BATCH_SESSION_IDLE_GRACE_SEC )); then
    batch_session_stop "max_session_age:${reason}"
    return 0
  fi
}

find_candidate() {
  local skip_orphans="${1:-1}"
  local orphan_path=""
  if [[ "$skip_orphans" == "1" && -f "$ORPHAN_FILE" ]]; then
    orphan_path="$ORPHAN_FILE"
  fi

  python3 - "$QUEUE_ROOT" "$CANDIDATE_FILE" "$orphan_path" "$RUN_INDEX_PATH" "$PROMOTION_PRE_RANK_TOPK" "$FULLSPAN_DECISION_STATE_FILE" "$LOW_YIELD_HOMOGENEOUS_FAIL_FRACTION" "$LOW_YIELD_HOMOGENEOUS_FAIL_MIN" "$FULLSPAN_CONFIRM_MIN_GROUPS" "$SELECTOR_STALLED_BUDGET_RATIO" "$READY_BUFFER_POOL_FILE" "$COLD_FAIL_STATE_FILE" "$READY_BUFFER_TARGET_DEPTH" "$FULLSPAN_POLICY_NAME" <<'PY'
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
    "METRICS_MISSING": "METRICS_MISSING",
    "TRADES_FAIL": "TRADES_FAIL",
    "PAIRS_FAIL": "PAIRS_FAIL",
    "DD_FAIL": "DD_FAIL",
    "ECONOMIC_FAIL": "ECONOMIC_FAIL",
    "STEP_FAIL": "STEP_FAIL",
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


def config_exists(path_raw):
    path_s = str(path_raw or '').strip()
    if not path_s:
        return False
    cfg = Path(path_s)
    if not cfg.is_absolute():
        cfg = app_root / cfg
    return cfg.exists()


def hard_gate_pass(row):
    if not row:
        return False, "metrics_missing"
    result = evaluate_row_hard_gates(row, CONTRACT_THRESHOLDS)
    if not result.passed:
        return False, str(result.reason or "METRICS_MISSING")
    return True, "PASS"


def score_fullspan_v1_like(sharpes, tails):
    if not sharpes or not tails:
        return None
    try:
        worst_robust = min(float(v) for v in sharpes)
        worst_step = min(float(v) for v in tails)
        q20 = sorted(float(v) for v in tails)[max(0, int((len(tails) - 1) * 0.20))] if tails else worst_step
        return score_fullspan_v1(
            worst_robust_sharpe=float(worst_robust),
            q_step_pnl=float(q20),
            worst_step_pnl=float(worst_step),
            initial_capital=float(INITIAL_CAPITAL),
            tail_q_soft_loss_pct=0.03,
            tail_worst_soft_loss_pct=0.10,
            tail_q_penalty=2.0,
            tail_worst_penalty=1.0,
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

try:
    SELECTOR_STALLED_BUDGET_RATIO = float(sys.argv[10])
except Exception:
    SELECTOR_STALLED_BUDGET_RATIO = 0.20
SELECTOR_STALLED_BUDGET_RATIO = max(0.0, min(1.0, SELECTOR_STALLED_BUDGET_RATIO))


queue_root = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
try:
    app_root = queue_root.parents[2]
except Exception:
    app_root = queue_root
opt_dir = app_root / "scripts" / "optimization"
if str(opt_dir) not in sys.path:
    sys.path.insert(0, str(opt_dir))
from fullspan_contract import FullspanThresholds, evaluate_row_hard_gates, row_worst_step_pnl, score_fullspan_v1

CONTRACT_THRESHOLDS = FullspanThresholds(
    min_trades=200.0,
    min_pairs=20.0,
    max_dd_pct=0.20,
    min_pnl=0.0,
    initial_capital=INITIAL_CAPITAL,
    max_worst_step_loss_pct=0.20,
)
orphan_file = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
run_index_path = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
pre_rank_top_k = int(sys.argv[5]) if len(sys.argv) > 5 and str(sys.argv[5]).strip() else 8
state_path = Path(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else None
ready_pool_path = Path(sys.argv[11]) if len(sys.argv) > 11 and sys.argv[11] else None
cold_fail_state_path = Path(sys.argv[12]) if len(sys.argv) > 12 and sys.argv[12] else None
ready_buffer_target_depth = int(sys.argv[13]) if len(sys.argv) > 13 and str(sys.argv[13]).strip() else 3
policy_version = str(sys.argv[14] if len(sys.argv) > 14 else "").strip() or "fullspan_v1"
state_by_queue = load_state(state_path)


def load_cold_fail_state(path: Path | None):
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    active = {}
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        queue = str(entry.get("queue") or "").strip()
        if not queue:
            continue
        active[queue] = entry
    return active


cold_fail_state = load_cold_fail_state(cold_fail_state_path)

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
    entries = []

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

        executable_pending = 0
        executable_planned = 0
        for row in rows:
            status = str(row.get('status') or '').strip().lower()
            if status == 'running':
                executable_pending += 1
                continue
            if status in {'planned', 'stalled', 'failed', 'error'} and config_exists(row.get('config_path')):
                executable_pending += 1
                if status == 'planned':
                    executable_planned += 1

        queue_abs = str(p)
        try:
            queue_rel = str(p.relative_to(app_root))
        except Exception:
            queue_rel = queue_abs
        mtime = float(p.stat().st_mtime)

        cold_entry = cold_fail_state.get(queue_rel) or cold_fail_state.get(queue_abs) or cold_fail_state.get('/' + queue_rel.lstrip('/'))
        if cold_entry:
            try:
                until_ts = float(cold_entry.get('until_ts') or 0.0)
            except Exception:
                until_ts = 0.0
            try:
                inserted_ts = float(cold_entry.get('inserted_ts') or 0.0)
            except Exception:
                inserted_ts = 0.0
            entry_policy = str(cold_entry.get('policy_version') or '').strip() or policy_version
            if entry_policy == policy_version and until_ts > now and inserted_ts > 0.0 and mtime <= inserted_ts:
                continue

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
            state_verdict in ('PROMOTE_PENDING_CONFIRM', 'PROMOTE_DEFER_CONFIRM')
            and state_strict_pass_count > 0
            and state_strict_run_groups >= FULLSPAN_CONFIRM_MIN_GROUPS
        )

        for _, bundle in by_base.items():
            h = bundle.get('holdout')
            s = bundle.get('stress')

            if h is None and s is None:
                unknown_configs += 1
                continue

            h_ok = True
            s_ok = True
            h_reason = ''
            s_reason = ''

            if h is None:
                h_ok = False
                h_reason = 'metrics_missing'
            else:
                ok, reason = hard_gate_pass(h)
                if not ok:
                    h_ok = False
                    h_reason = reason

            if s is None:
                s_ok = False
                s_reason = 'metrics_missing'
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
                h_tail = row_worst_step_pnl(h)
                s_tail = row_worst_step_pnl(s)
                tails = [v for v in (h_tail, s_tail) if v is not None]
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

        strict_gate_status = 'FULLSPAN_PREFILTER_PASSED'
        strict_gate_reason = ''
        if state_verdict == 'REJECT' or state_strict_status == 'FULLSPAN_PREFILTER_REJECT':
            promotion_potential = 'REJECT'
            gate_status = 'HARD_FAIL'
            strict_gate_status = 'FULLSPAN_PREFILTER_REJECT'
            gate_reason = canonical_state_reason(state_strict_reason or 'METRICS_MISSING')
        elif total_configs == 0:
            promotion_potential = 'UNKNOWN'
            gate_status = 'OPEN'
            strict_gate_status = 'FULLSPAN_PREFILTER_UNKNOWN'
            gate_reason = 'insufficient_history'
        elif pass_configs == 0 and fail_configs > 0:
            promotion_potential = 'REJECT'
            gate_status = 'HARD_FAIL'
            strict_gate_status = 'FULLSPAN_PREFILTER_REJECT'
            dominant_reason = fail_reasons[0] if fail_reasons else ''
            if by_state_reason and fail_configs >= LOW_YIELD_HOMOGENEOUS_MIN:
                try:
                    dominant_reason_name, dominant_reason_count = max(by_state_reason.items(), key=lambda item: item[1])
                    threshold = max(LOW_YIELD_HOMOGENEOUS_MIN, int(fail_configs * LOW_YIELD_HOMOGENEOUS_FRACTION))
                    if dominant_reason_count >= threshold:
                        dominant_reason = f'LOW_YIELD_HOMOGENEOUS_{dominant_reason_name}'
                except Exception:
                    pass
            gate_reason = dominant_reason or 'METRICS_MISSING'
        else:
            promotion_potential = 'POSSIBLE'
            gate_status = 'OPEN'
            if fail_configs > 0:
                strict_gate_status = 'FULLSPAN_PREFILTER_DEGRADED'
            gate_reason = 'history_probe'

        if strict_gate_status in ('FULLSPAN_PREFILTER_REJECT', 'HARD_FAIL') and gate_status == 'HARD_FAIL':
            strict_gate_reason = gate_reason
        if strict_gate_status in ('FULLSPAN_PREFILTER_REJECT', 'HARD_FAIL') and strict_gate_reason:
            strict_gate_reason = canonical_state_reason(strict_gate_reason)
        if gate_status == 'OPEN' and strict_gate_status == 'FULLSPAN_PREFILTER_PASSED':
            strict_gate_reason = ''

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
        pre_rank_score += 20.0 if strict_gate_status == 'FULLSPAN_PREFILTER_PASSED' else 0.0
        pre_rank_score += min(max(state_strict_pass_count, 0), 3) * 6.0
        pre_rank_score += min(max(state_strict_run_groups, 0), 2) * 10.0
        pre_rank_score += min(max(state_confirm_count, 0), 2) * 12.0

        if state_verdict == 'PROMOTE_ELIGIBLE':
            pre_rank_score += 10.0
        elif confirm_fastlane_ready:
            pre_rank_score += 20.0
            if confirm_pending_since_epoch > 0:
                age_hours = max(0.0, (now - confirm_pending_since_epoch) / 3600.0)
                pre_rank_score += min(age_hours, 24.0) * 1.5

        if fail_configs > 0 and by_state_reason:
            dominant_reason_count = max(by_state_reason.values())
            if dominant_reason_count >= max(2, int(0.7 * max(fail_configs, 1))):
                pre_rank_score -= 6.0

        urgency = (stalled * 100.0) + (running * 20.0) + (age_min * 0.1) + (1.0 if pending > 0 else 0.0)
        if gate_status == 'HARD_FAIL' and promotion_potential == 'REJECT':
            pre_rank_score -= 10000

        effective_planned_count = int(executable_planned if executable_planned > 0 else planned)
        stalled_share = float(stalled) / float(max(1, pending))
        queue_yield_score = float(pass_configs) / float(max(1, total_configs))
        completed_with_metrics = 0
        if pass_configs > 0:
            completed_with_metrics = pass_configs
        else:
            for bundle in by_base.values():
                for key in ('holdout', 'stress'):
                    row = bundle.get(key)
                    if row is not None and is_true(row.get('metrics_present')):
                        completed_with_metrics += 1
        started_window = max(1, planned + running + stalled + failed + completed)
        recent_yield = float(completed_with_metrics) / float(started_window)
        pre_rank_score += recent_yield * 25.0
        if fail_configs > 0 and pass_configs == 0:
            queue_yield_score -= min(1.0, float(fail_configs) / float(max(1, total_configs)))

        if effective_planned_count <= 0 and stalled > 0:
            over_budget = max(0.0, stalled_share - SELECTOR_STALLED_BUDGET_RATIO)
            if over_budget > 0.0:
                pre_rank_score -= (40.0 + over_budget * 220.0)
                queue_yield_score -= min(0.8, over_budget)

        entries.append({
            'queue_rel': queue_rel,
            'planned': planned,
            'running': running,
            'stalled': stalled,
            'failed': failed,
            'completed': completed,
            'total': total,
            'urgency': urgency,
            'mtime': int(mtime),
            'promotion_potential': promotion_potential,
            'gate_status': gate_status,
            'gate_reason': gate_reason,
            'pre_rank_score': pre_rank_score,
            'strict_gate_status': strict_gate_status,
            'strict_gate_reason': strict_gate_reason,
            'effective_planned_count': effective_planned_count,
            'stalled_share': stalled_share,
            'queue_yield_score': queue_yield_score,
            'recent_yield': recent_yield,
            'executable_pending': executable_pending,
        })

    header = 'queue,planned,running,stalled,failed,completed,total,urgency,mtime,promotion_potential,gate_status,gate_reason,pre_rank_score,strict_gate_status,strict_gate_reason,effective_planned_count,stalled_share,queue_yield_score,recent_yield\n'
    pool_header = header[:-1] + ',executable_pending\n'
    with out_csv.open('w', encoding='utf-8', newline='') as f:
        f.write(header)
        if not entries:
            if ready_pool_path is not None:
                ready_pool_path.parent.mkdir(parents=True, exist_ok=True)
                with ready_pool_path.open('w', encoding='utf-8', newline='') as pool_handle:
                    pool_handle.write(pool_header)
            raise SystemExit(0)

        has_effective_planned = any(int(item.get('effective_planned_count', 0)) > 0 for item in entries)
        pool = entries
        if has_effective_planned:
            planned_only = [item for item in entries if int(item.get('effective_planned_count', 0)) > 0]
            if planned_only:
                pool = planned_only

        ranked = []
        for item in pool:
            executable_rank = 1 if int(item.get('executable_pending', 0)) > 0 else 0
            planned_rank = 1 if int(item.get('effective_planned_count', 0)) > 0 else 0
            potential_rank = {'POSSIBLE': 2, 'UNKNOWN': 1, 'REJECT': 0}.get(str(item.get('promotion_potential', 'UNKNOWN')), 1)
            ranked.append((
                -planned_rank,
                -executable_rank,
                -potential_rank,
                -float(item.get('queue_yield_score', 0.0)),
                -float(item.get('pre_rank_score', 0.0)),
                -float(item.get('urgency', 0.0)),
                -int(item.get('mtime', 0)),
                str(item.get('queue_rel') or ''),
                item,
            ))

        ranked.sort()
        ranked = ranked[:max(1, pre_rank_top_k, ready_buffer_target_depth)]
        top_entries = [item[-1] for item in ranked]
        winner = top_entries[0]

        f.write(
            '{queue},{planned},{running},{stalled},{failed},{completed},{total},{urgency:.3f},{mtime},{potential},{gate_status},{gate_reason},{pre_rank:.3f},{strict_status},{strict_reason},{effective_planned},{stalled_share:.3f},{yield_score:.3f},{recent_yield:.3f}\n'.format(
                queue=winner['queue_rel'],
                planned=int(winner['planned']),
                running=int(winner['running']),
                stalled=int(winner['stalled']),
                failed=int(winner['failed']),
                completed=int(winner['completed']),
                total=int(winner['total']),
                urgency=float(winner['urgency']),
                mtime=int(winner['mtime']),
                potential=str(winner['promotion_potential']),
                gate_status=str(winner['gate_status']),
                gate_reason=str(winner['gate_reason']),
                pre_rank=float(winner['pre_rank_score']),
                strict_status=str(winner['strict_gate_status']),
                strict_reason=str(winner['strict_gate_reason']),
                effective_planned=int(winner['effective_planned_count']),
                stalled_share=float(winner['stalled_share']),
                yield_score=float(winner['queue_yield_score']),
                recent_yield=float(winner.get('recent_yield', 0.0)),
            )
        )

    if ready_pool_path is not None:
        ready_pool_path.parent.mkdir(parents=True, exist_ok=True)
        with ready_pool_path.open('w', encoding='utf-8', newline='') as pool_handle:
            pool_handle.write(pool_header)
            for item in top_entries[:max(1, ready_buffer_target_depth)]:
                pool_handle.write(
                    '{queue},{planned},{running},{stalled},{failed},{completed},{total},{urgency:.3f},{mtime},{potential},{gate_status},{gate_reason},{pre_rank:.3f},{strict_status},{strict_reason},{effective_planned},{stalled_share:.3f},{yield_score:.3f},{recent_yield:.3f},{executable_pending}\n'.format(
                        queue=item['queue_rel'],
                        planned=int(item['planned']),
                        running=int(item['running']),
                        stalled=int(item['stalled']),
                        failed=int(item['failed']),
                        completed=int(item['completed']),
                        total=int(item['total']),
                        urgency=float(item['urgency']),
                        mtime=int(item['mtime']),
                        potential=str(item['promotion_potential']),
                        gate_status=str(item['gate_status']),
                        gate_reason=str(item['gate_reason']),
                        pre_rank=float(item['pre_rank_score']),
                        strict_status=str(item['strict_gate_status']),
                        strict_reason=str(item['strict_gate_reason']),
                        effective_planned=int(item['effective_planned_count']),
                        stalled_share=float(item['stalled_share']),
                        yield_score=float(item['queue_yield_score']),
                        recent_yield=float(item.get('recent_yield', 0.0)),
                        executable_pending=int(item.get('executable_pending', 0)),
                    )
                )


emit_scores()
PY
}

fallback_pending_candidate() {
  local orphan_path="${1:-}"
  local skip_queue="${2:-}"
  local state_path="${3:-}"
  python3 - "$QUEUE_ROOT" "$CANDIDATE_FILE" "$orphan_path" "$skip_queue" "$state_path" <<'PY'
import csv
from collections import Counter
import json
from pathlib import Path
import time
import sys

queue_root = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
orphan_file = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
skip_queue = str(sys.argv[4] if len(sys.argv) > 4 else "").strip()
state_path = Path(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None
try:
    app_root = queue_root.parents[2]
except Exception:
    app_root = queue_root


def config_exists(path_raw):
    path_s = str(path_raw or '').strip()
    if not path_s:
        return False
    cfg = Path(path_s)
    if not cfg.is_absolute():
        cfg = app_root / cfg
    return cfg.exists()

now = time.time()
orphan = {}
if orphan_file and orphan_file.exists():
    try:
        with orphan_file.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                q = str(row.get('queue') or '').strip()
                if not q:
                    continue
                try:
                    until = float((row.get('until_ts') or '').strip() or 0)
                except Exception:
                    until = 0
                orphan[q] = until
    except Exception:
        orphan = {}

state_reject = set()
if state_path and state_path.exists():
    try:
        state = json.loads(state_path.read_text(encoding='utf-8'))
        queues = state.get('queues', {}) if isinstance(state, dict) else {}
        if isinstance(queues, dict):
            for qrel, entry in queues.items():
                if not isinstance(entry, dict):
                    continue
                verdict = str(entry.get('promotion_verdict') or '').strip().upper()
                if verdict == 'REJECT':
                    state_reject.add(str(qrel).strip())
    except Exception:
        state_reject = set()

best = None
for p in sorted(queue_root.rglob('run_queue.csv')):
    if '/rollup/' in str(p) or '/.autonomous/' in str(p):
        continue
    try:
        queue_rel = str(p.relative_to(app_root))
    except Exception:
        queue_rel = str(p)
    queue_abs = str(p)
    if skip_queue and skip_queue in {queue_rel, queue_abs, '/' + queue_rel.lstrip('/')}:
        continue
    if queue_rel in state_reject or queue_abs in state_reject:
        continue
    orphan_until = None
    for key in (queue_rel, queue_abs, '/' + queue_rel.lstrip('/')):
        if key in orphan:
            orphan_until = orphan.get(key, 0.0)
            break
    if orphan_until is not None and now < float(orphan_until or 0.0):
        continue

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

    executable_pending = 0
    for row in rows:
        status = str(row.get('status') or '').strip().lower()
        if status == 'running':
            executable_pending += 1
            continue
        if status in {'planned', 'stalled', 'failed', 'error'} and config_exists(row.get('config_path')):
            executable_pending += 1

    mtime = int(p.stat().st_mtime)
    age_min = max(0.0, (time.time() - mtime) / 60.0)
    urgency = (stalled * 100.0) + (running * 20.0) + (age_min * 0.1) + 1.0

    # Prefer queues with planned work to avoid spinning on already fail-closed stalled tails.
    planned_priority = 1 if planned > 0 else 0
    row = (
        1 if executable_pending > 0 else 0,
        executable_pending,
        planned_priority,
        pending,
        stalled,
        running,
        -mtime,
        queue_rel,
        planned,
        running,
        stalled,
        failed,
        completed,
        total,
        urgency,
        mtime,
    )
    if best is None or row > best:
        best = row

if best is None:
    raise SystemExit(1)

_, _, _, _, _, _, _, queue, planned, running, stalled, failed, completed, total, urgency, mtime = best
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open('w', encoding='utf-8', newline='') as f:
    f.write('queue,planned,running,stalled,failed,completed,total,urgency,mtime,promotion_potential,gate_status,gate_reason,pre_rank_score,strict_gate_status,strict_gate_reason,effective_planned_count,stalled_share,queue_yield_score,recent_yield\n')
    stalled_share = float(stalled) / float(max((planned + running + stalled + failed), 1))
    queue_yield_score = float(planned) * 4.0 - (stalled_share * 20.0)
    recent_yield = float(completed) / float(max((planned + running + stalled + failed + completed), 1))
    f.write(f"{queue},{planned},{running},{stalled},{failed},{completed},{total},{urgency:.3f},{mtime},POSSIBLE,OPEN,fallback_pending,0.000,FULLSPAN_PREFILTER_UNKNOWN,,{int(planned)},{stalled_share:.6f},{queue_yield_score:.3f},{recent_yield:.3f}\n")
PY
}

ready_buffer_depth() {
  python3 - "$READY_BUFFER_STATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit(0)
try:
    data = json.loads(path.read_text(encoding='utf-8'))
except Exception:
    print(0)
    raise SystemExit(0)
entries = data.get("entries", []) if isinstance(data, dict) else []
if not isinstance(entries, list):
    print(0)
    raise SystemExit(0)
count = 0
for entry in entries:
    if isinstance(entry, dict) and not bool(entry.get("claimed")):
        count += 1
print(count)
PY
}

ready_buffer_state_value() {
  local key="$1"
  python3 - "$READY_BUFFER_STATE_FILE" "$key" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = str(sys.argv[2] or "").strip()
if not path.exists() or not key:
    print("")
    raise SystemExit(0)
try:
    data = json.loads(path.read_text(encoding='utf-8'))
except Exception:
    print("")
    raise SystemExit(0)
value = data.get(key) if isinstance(data, dict) else ""
if isinstance(value, bool):
    print("1" if value else "0")
elif isinstance(value, (int, float)):
    print(value)
elif isinstance(value, str):
    print(value)
else:
    print("")
PY
}

current_planner_policy_hash() {
  python3 - "$SEARCH_DIRECTOR_DIRECTIVE_FILE" "$YIELD_GOVERNOR_STATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

for candidate in sys.argv[1:]:
    path = Path(candidate)
    if not path.exists():
        continue
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        continue
    if not isinstance(payload, dict):
        continue
    for key in ("policy_hash", "policy-hash"):
        value = str(payload.get(key) or "").strip()
        if value:
            print(value)
            raise SystemExit(0)
print("")
PY
}

is_hot_standby_enabled() {
  is_truthy "$VPS_HOT_STANDBY_ENABLE"
}

confirm_fastlane_pending_count() {
  python3 - "$FULLSPAN_DECISION_STATE_FILE" "$ROOT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
root_dir = Path(sys.argv[2])
if not state_path.exists():
    print(0)
    raise SystemExit(0)
try:
    payload = json.loads(state_path.read_text(encoding='utf-8'))
except Exception:
    print(0)
    raise SystemExit(0)
queues = payload.get("queues", {}) if isinstance(payload, dict) else {}
if not isinstance(queues, dict):
    print(0)
    raise SystemExit(0)
pending = 0
for entry in queues.values():
    if not isinstance(entry, dict):
        continue
    queue_rel = str(entry.get("confirm_fastlane_queue_rel") or "").strip()
    if not queue_rel:
        continue
    queue_path = root_dir / queue_rel
    if not queue_path.exists():
        continue
    try:
        rows = list(csv.DictReader(queue_path.open(newline='', encoding='utf-8')))
    except Exception:
        continue
    for row in rows:
        status = str(row.get("status") or "").strip().lower()
        if status in {"planned", "running", "stalled", "failed", "error"}:
            pending += 1
            break
print(pending)
PY
}

ready_buffer_policy_hash() {
  local planner_policy_hash
  planner_policy_hash="$(current_planner_policy_hash)"
  python3 - "$READY_BUFFER_POOL_FILE" "$GATE_SURROGATE_STATE_FILE" "$FULLSPAN_DECISION_STATE_FILE" "$FULLSPAN_POLICY_NAME" "$READY_BUFFER_TARGET_DEPTH" "$READY_BUFFER_REFILL_THRESHOLD" "$planner_policy_hash" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

pool_path = Path(sys.argv[1])
surrogate_path = Path(sys.argv[2])
fullspan_path = Path(sys.argv[3])
policy_version = str(sys.argv[4] or "").strip() or "fullspan_v1"
try:
    target_depth = max(1, int(float(sys.argv[5] or 3)))
except Exception:
    target_depth = 3
try:
    refill_threshold = max(1, int(float(sys.argv[6] or 2)))
except Exception:
    refill_threshold = 2
planner_policy_hash = str(sys.argv[7] or "").strip()

pool_rows = []
if pool_path.exists():
    try:
        for row in csv.DictReader(pool_path.open(newline="", encoding="utf-8")):
            if len(pool_rows) >= max(8, target_depth):
                break
            queue = str(row.get("queue") or "").strip()
            if not queue:
                continue
            pool_rows.append(
                {
                    "queue": queue,
                    "mtime": int(float(row.get("mtime") or 0)),
                    "promotion_potential": str(row.get("promotion_potential") or "").strip(),
                    "gate_status": str(row.get("gate_status") or "").strip(),
                    "strict_gate_status": str(row.get("strict_gate_status") or "").strip(),
                    "pre_rank_score": str(row.get("pre_rank_score") or "").strip(),
                    "executable_pending": int(float(row.get("executable_pending") or 0)),
                }
            )
    except Exception:
        pool_rows = []

payload = {
    "policy_version": policy_version,
    "target_depth": target_depth,
    "refill_threshold": refill_threshold,
    "planner_policy_hash": planner_policy_hash,
    "pool_rows": pool_rows,
    "pool_mtime": int(pool_path.stat().st_mtime) if pool_path.exists() else 0,
    "surrogate_mtime": int(surrogate_path.stat().st_mtime) if surrogate_path.exists() else 0,
    "fullspan_mtime": int(fullspan_path.stat().st_mtime) if fullspan_path.exists() else 0,
}
body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
print(hashlib.sha256(body.encode("utf-8")).hexdigest())
PY
}

cold_fail_active_count() {
  python3 - "$COLD_FAIL_STATE_FILE" "$FULLSPAN_POLICY_NAME" <<'PY'
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
policy = str(sys.argv[2] or "").strip()
now = time.time()
if not path.exists():
    print(0)
    raise SystemExit(0)
try:
    data = json.loads(path.read_text(encoding='utf-8'))
except Exception:
    print(0)
    raise SystemExit(0)
entries = data.get("entries", []) if isinstance(data, dict) else []
if not isinstance(entries, list):
    print(0)
    raise SystemExit(0)
count = 0
for entry in entries:
    if not isinstance(entry, dict):
        continue
    entry_policy = str(entry.get("policy_version") or "").strip()
    if entry_policy and entry_policy != policy:
        continue
    try:
        until_ts = float(entry.get("until_ts") or 0.0)
    except Exception:
        until_ts = 0.0
    if until_ts > now:
        count += 1
print(count)
PY
}

cold_fail_state_add() {
  local queue_rel="$1"
  local gate_reason="$2"
  local until_ts
  local now_ts
  now_ts="$(date +%s)"
  until_ts="$(( now_ts + HARD_FAIL_COLD_TTL_SEC ))"
  python3 - "$COLD_FAIL_STATE_FILE" "$queue_rel" "$gate_reason" "$now_ts" "$until_ts" "$FULLSPAN_POLICY_NAME" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
queue = str(sys.argv[2] or "").strip()
gate_reason = str(sys.argv[3] or "").strip()
inserted_ts = float(sys.argv[4] or 0.0)
until_ts = float(sys.argv[5] or 0.0)
policy_version = str(sys.argv[6] or "").strip() or "fullspan_v1"

payload = {"ts": "", "policy_version": policy_version, "entries": []}
if path.exists():
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        payload = {"ts": "", "policy_version": policy_version, "entries": []}
if not isinstance(payload, dict):
    payload = {"ts": "", "policy_version": policy_version, "entries": []}
entries = payload.get("entries", [])
if not isinstance(entries, list):
    entries = []

fresh = []
for entry in entries:
    if not isinstance(entry, dict):
        continue
    if str(entry.get("queue") or "").strip() == queue:
        continue
    fresh.append(entry)

fresh.append(
    {
        "queue": queue,
        "gate_reason": gate_reason,
        "source_verdict": "REJECT",
        "inserted_ts": inserted_ts,
        "until_ts": until_ts,
        "policy_version": policy_version,
    }
)
payload["entries"] = fresh[-5000:]
payload["policy_version"] = policy_version
payload["ts"] = ""
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
PY
}

ready_buffer_refresh() {
  local exclude_queue="${1:-}"
  local py_bin="$ROOT_DIR/.venv/bin/python"
  local policy_hash
  [[ -x "$py_bin" ]] || py_bin="$(command -v python3)"
  if [[ -z "$py_bin" || ! -f "$READY_BUFFER_POOL_FILE" ]]; then
    return 1
  fi
  policy_hash="$(ready_buffer_policy_hash)"

  local refresh_needed="1"
  refresh_needed="$(python3 - "$GATE_SURROGATE_STATE_FILE" "$READY_BUFFER_POOL_FILE" <<'PY'
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

state_path = Path(sys.argv[1])
pool_path = Path(sys.argv[2])
fresh_sec = 90
if not pool_path.exists():
    print("0")
    raise SystemExit(0)
if not state_path.exists():
    print("1")
    raise SystemExit(0)
try:
    data = json.loads(state_path.read_text(encoding='utf-8'))
except Exception:
    print("1")
    raise SystemExit(0)
ts = str(data.get("ts") or "").strip()
age_sec = fresh_sec + 1
if ts:
    try:
        parsed = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        age_sec = max(0, int(time.time() - int(parsed.timestamp())))
    except Exception:
        age_sec = fresh_sec + 1
state_mtime = state_path.stat().st_mtime if state_path.exists() else 0.0
pool_mtime = pool_path.stat().st_mtime if pool_path.exists() else 0.0
print("1" if age_sec > fresh_sec or pool_mtime > state_mtime else "0")
PY
)"
  if [[ "$refresh_needed" == "1" ]]; then
    if "$py_bin" "$ROOT_DIR/scripts/optimization/gate_surrogate_agent.py" --root "$ROOT_DIR" >/dev/null 2>&1; then
      "$py_bin" "$ROOT_DIR/scripts/optimization/search_director_agent.py" --root "$ROOT_DIR" >/dev/null 2>&1 || true
      log "surrogate_gate_refresh queue=global status=ok reason=ready_buffer"
    else
      log "surrogate_gate_refresh queue=global status=failed reason=ready_buffer"
    fi
  fi

  python3 - "$READY_BUFFER_POOL_FILE" "$READY_BUFFER_STATE_FILE" "$GATE_SURROGATE_STATE_FILE" "$ORPHAN_FILE" "$FULLSPAN_DECISION_STATE_FILE" "$exclude_queue" "$READY_BUFFER_TARGET_DEPTH" "$READY_BUFFER_REFILL_THRESHOLD" "$ROOT_DIR" "$FULLSPAN_POLICY_NAME" "$policy_hash" <<'PY'
import csv
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

pool_path = Path(sys.argv[1])
state_path = Path(sys.argv[2])
surrogate_path = Path(sys.argv[3])
orphan_path = Path(sys.argv[4])
fullspan_path = Path(sys.argv[5])
exclude_queue = str(sys.argv[6] or "").strip()
target_depth = max(1, int(float(sys.argv[7] or 3)))
refill_threshold = max(1, int(float(sys.argv[8] or 2)))
root_dir = Path(sys.argv[9])
policy_version = str(sys.argv[10] or "").strip() or "fullspan_v1"
policy_hash = str(sys.argv[11] or "").strip()

def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return default

def queue_pending(queue_rel: str) -> int:
    queue_path = root_dir / queue_rel
    if not queue_path.exists():
        return 0
    try:
        rows = list(csv.DictReader(queue_path.open(newline='', encoding='utf-8')))
    except Exception:
        return 0
    pending = 0
    for row in rows:
        status = str(row.get("status") or "").strip().lower()
        if status in {"planned", "running", "stalled", "failed", "error"}:
            pending += 1
    return pending

rows = []
if pool_path.exists():
    try:
        rows = list(csv.DictReader(pool_path.open(newline='', encoding='utf-8')))
    except Exception:
        rows = []

surrogate = load_json(surrogate_path, {})
surrogate_queues = surrogate.get("queues", {}) if isinstance(surrogate, dict) else {}
orphans = {}
if orphan_path.exists():
    try:
        for row in csv.DictReader(orphan_path.open(newline='', encoding='utf-8')):
            queue = str(row.get("queue") or "").strip()
            if not queue:
                continue
            try:
                orphans[queue] = float(row.get("until_ts") or 0.0)
            except Exception:
                orphans[queue] = 0.0
    except Exception:
        orphans = {}
fullspan = load_json(fullspan_path, {})
fullspan_queues = fullspan.get("queues", {}) if isinstance(fullspan, dict) else {}
now_ts = datetime.now(timezone.utc).timestamp()

entries = []
for row in rows:
    queue = str(row.get("queue") or "").strip()
    if not queue or queue == exclude_queue:
        continue
    if queue_pending(queue) <= 0:
        continue
    if queue in orphans and float(orphans.get(queue) or 0.0) > now_ts:
        continue
    state_entry = fullspan_queues.get(queue, {})
    if isinstance(state_entry, dict):
        verdict = str(state_entry.get("promotion_verdict") or "").strip().upper()
        if verdict == "REJECT":
            continue
    surrogate_entry = surrogate_queues.get(queue, {})
    if not surrogate_entry and (root_dir / queue).exists():
        surrogate_entry = surrogate_queues.get(str((root_dir / queue).resolve()), {})
    decision = str((surrogate_entry or {}).get("decision") or "").strip().lower()
    reason = str((surrogate_entry or {}).get("reason") or "").strip()
    if decision != "allow":
        continue
    queue_path = root_dir / queue
    if not queue_path.exists():
        continue
    queue_file_mtime = int(queue_path.stat().st_mtime)
    row_mtime = int(float(row.get("mtime") or 0))
    if row_mtime > 0 and queue_file_mtime > row_mtime:
        continue
    entry = {
        "queue": queue,
        "planned": int(float(row.get("planned") or 0)),
        "running": int(float(row.get("running") or 0)),
        "stalled": int(float(row.get("stalled") or 0)),
        "failed": int(float(row.get("failed") or 0)),
        "completed": int(float(row.get("completed") or 0)),
        "total": int(float(row.get("total") or 0)),
        "urgency": float(row.get("urgency") or 0.0),
        "mtime": int(float(row.get("mtime") or 0)),
        "promotion_potential": str(row.get("promotion_potential") or "POSSIBLE"),
        "gate_status": str(row.get("gate_status") or "OPEN"),
        "gate_reason": str(row.get("gate_reason") or ""),
        "pre_rank_score": float(row.get("pre_rank_score") or 0.0),
        "strict_gate_status": str(row.get("strict_gate_status") or ""),
        "strict_gate_reason": str(row.get("strict_gate_reason") or ""),
        "effective_planned_count": int(float(row.get("effective_planned_count") or 0)),
        "stalled_share": float(row.get("stalled_share") or 0.0),
        "queue_yield_score": float(row.get("queue_yield_score") or 0.0),
        "recent_yield": float(row.get("recent_yield") or 0.0),
        "executable_pending": int(float(row.get("executable_pending") or 0)),
        "surrogate_decision": decision,
        "surrogate_reason": reason,
        "queue_file_mtime": queue_file_mtime,
        "claimed": False,
        "claim_ts": 0,
    }
    entries.append(entry)
    if len(entries) >= target_depth:
        break

payload = {
    "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "built_epoch": int(datetime.now(timezone.utc).timestamp()),
    "policy_version": policy_version,
    "policy_hash": policy_hash,
    "target_depth": target_depth,
    "refill_threshold": refill_threshold,
    "ready_count": len(entries),
    "source_pool_mtime": int(pool_path.stat().st_mtime) if pool_path.exists() else 0,
    "source_surrogate_mtime": int(surrogate_path.stat().st_mtime) if surrogate_path.exists() else 0,
    "source_fullspan_mtime": int(fullspan_path.stat().st_mtime) if fullspan_path.exists() else 0,
    "entries": entries,
}
state_path.parent.mkdir(parents=True, exist_ok=True)
state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
PY

  local depth
  depth="$(ready_buffer_depth)"
  fullspan_state_metric_set "ready_buffer_depth" "${depth:-0}"
  fullspan_state_metric_set "ready_buffer_refill_threshold" "$READY_BUFFER_REFILL_THRESHOLD"
  fullspan_state_metric_set "ready_buffer_target_depth" "$READY_BUFFER_TARGET_DEPTH"
  log "ready_buffer_refresh depth=${depth:-0} exclude=${exclude_queue:-none}"
  return 0
}

ready_buffer_emit_candidate() {
  local exclude_queue="${1:-}"
  local rc=0
  local policy_hash
  policy_hash="$(ready_buffer_policy_hash)"
  python3 - "$READY_BUFFER_STATE_FILE" "$CANDIDATE_FILE" "$exclude_queue" "$READY_BUFFER_POOL_FILE" "$GATE_SURROGATE_STATE_FILE" "$FULLSPAN_DECISION_STATE_FILE" "$ROOT_DIR" "$READY_BUFFER_MAX_AGE_SEC" "$policy_hash" <<'PY'
import csv
import json
import sys
import time
from pathlib import Path

state_path = Path(sys.argv[1])
candidate_path = Path(sys.argv[2])
exclude_queue = str(sys.argv[3] or "").strip()
pool_path = Path(sys.argv[4])
surrogate_path = Path(sys.argv[5])
fullspan_path = Path(sys.argv[6])
root_dir = Path(sys.argv[7])
try:
    max_age_sec = max(30, int(float(sys.argv[8] or 900)))
except Exception:
    max_age_sec = 900
expected_hash = str(sys.argv[9] or "").strip()

header = [
    "queue",
    "planned",
    "running",
    "stalled",
    "failed",
    "completed",
    "total",
    "urgency",
    "mtime",
    "promotion_potential",
    "gate_status",
    "gate_reason",
    "pre_rank_score",
    "strict_gate_status",
    "strict_gate_reason",
    "effective_planned_count",
    "stalled_share",
    "queue_yield_score",
    "recent_yield",
]

if not state_path.exists():
    raise SystemExit(1)
try:
    data = json.loads(state_path.read_text(encoding='utf-8'))
except Exception:
    raise SystemExit(1)
policy_hash = str(data.get("policy_hash") or "").strip()
if expected_hash and policy_hash and policy_hash != expected_hash:
    raise SystemExit(23)
try:
    built_epoch = int(float(data.get("built_epoch") or 0))
except Exception:
    built_epoch = 0
if built_epoch > 0 and int(time.time()) - built_epoch > max_age_sec:
    raise SystemExit(24)
entries = data.get("entries", []) if isinstance(data, dict) else []
if not isinstance(entries, list):
    raise SystemExit(1)
selected = None
stale_skips = 0
for entry in entries:
    if not isinstance(entry, dict):
        continue
    queue = str(entry.get("queue") or "").strip()
    if not queue or queue == exclude_queue or bool(entry.get("claimed")):
        continue
    queue_path = root_dir / queue
    if not queue_path.exists():
        stale_skips += 1
        continue
    try:
        queue_mtime = int(queue_path.stat().st_mtime)
    except Exception:
        queue_mtime = 0
    try:
        state_queue_mtime = int(float(entry.get("queue_file_mtime") or 0))
    except Exception:
        state_queue_mtime = 0
    if state_queue_mtime > 0 and queue_mtime > state_queue_mtime:
        stale_skips += 1
        continue
    pending = 0
    try:
        for row in csv.DictReader(queue_path.open(newline="", encoding="utf-8")):
            status = str(row.get("status") or "").strip().lower()
            if status in {"planned", "running", "stalled", "failed", "error"}:
                pending += 1
    except Exception:
        pending = 0
    if pending <= 0:
        stale_skips += 1
        continue
    selected = entry
    break
if selected is None:
    raise SystemExit(24 if stale_skips > 0 else 1)
selected["claimed"] = True
selected["claim_ts"] = int(time.time())
state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
candidate_path.parent.mkdir(parents=True, exist_ok=True)
with candidate_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=header)
    writer.writeheader()
    writer.writerow(
        {
            "queue": selected.get("queue", ""),
            "planned": selected.get("planned", 0),
            "running": selected.get("running", 0),
            "stalled": selected.get("stalled", 0),
            "failed": selected.get("failed", 0),
            "completed": selected.get("completed", 0),
            "total": selected.get("total", 0),
            "urgency": f"{float(selected.get('urgency', 0.0)):.3f}",
            "mtime": selected.get("mtime", 0),
            "promotion_potential": selected.get("promotion_potential", "POSSIBLE"),
            "gate_status": selected.get("gate_status", "OPEN"),
            "gate_reason": selected.get("gate_reason", ""),
            "pre_rank_score": f"{float(selected.get('pre_rank_score', 0.0)):.3f}",
            "strict_gate_status": selected.get("strict_gate_status", ""),
            "strict_gate_reason": selected.get("strict_gate_reason", ""),
            "effective_planned_count": selected.get("effective_planned_count", 0),
            "stalled_share": f"{float(selected.get('stalled_share', 0.0)):.6f}",
            "queue_yield_score": f"{float(selected.get('queue_yield_score', 0.0)):.3f}",
            "recent_yield": f"{float(selected.get('recent_yield', 0.0)):.3f}",
        }
)
print(str(selected.get("queue") or "").strip())
PY
  rc=$?
  if (( rc == 23 )); then
    fullspan_state_metric_inc "ready_buffer_policy_mismatch_count" 1
    log "ready_buffer_policy_mismatch expected=$policy_hash"
    ready_buffer_refresh "$exclude_queue" || true
    python3 - "$READY_BUFFER_STATE_FILE" "$CANDIDATE_FILE" "$exclude_queue" "$READY_BUFFER_POOL_FILE" "$GATE_SURROGATE_STATE_FILE" "$FULLSPAN_DECISION_STATE_FILE" "$ROOT_DIR" "$READY_BUFFER_MAX_AGE_SEC" "$policy_hash" <<'PY'
import csv
import json
import sys
import time
from pathlib import Path

state_path = Path(sys.argv[1])
candidate_path = Path(sys.argv[2])
exclude_queue = str(sys.argv[3] or "").strip()
root_dir = Path(sys.argv[7])
try:
    max_age_sec = max(30, int(float(sys.argv[8] or 900)))
except Exception:
    max_age_sec = 900
expected_hash = str(sys.argv[9] or "").strip()

header = [
    "queue",
    "planned",
    "running",
    "stalled",
    "failed",
    "completed",
    "total",
    "urgency",
    "mtime",
    "promotion_potential",
    "gate_status",
    "gate_reason",
    "pre_rank_score",
    "strict_gate_status",
    "strict_gate_reason",
    "effective_planned_count",
    "stalled_share",
    "queue_yield_score",
    "recent_yield",
]

if not state_path.exists():
    raise SystemExit(1)
data = json.loads(state_path.read_text(encoding='utf-8'))
if expected_hash and str(data.get("policy_hash") or "").strip() != expected_hash:
    raise SystemExit(23)
try:
    built_epoch = int(float(data.get("built_epoch") or 0))
except Exception:
    built_epoch = 0
if built_epoch > 0 and int(time.time()) - built_epoch > max_age_sec:
    raise SystemExit(24)
entries = data.get("entries", []) if isinstance(data, dict) else []
selected = None
for entry in entries:
    if not isinstance(entry, dict):
        continue
    queue = str(entry.get("queue") or "").strip()
    if not queue or queue == exclude_queue or bool(entry.get("claimed")):
        continue
    queue_path = root_dir / queue
    if not queue_path.exists():
        continue
    pending = 0
    for row in csv.DictReader(queue_path.open(newline="", encoding="utf-8")):
        status = str(row.get("status") or "").strip().lower()
        if status in {"planned", "running", "stalled", "failed", "error"}:
            pending += 1
    if pending <= 0:
        continue
    selected = entry
    break
if selected is None:
    raise SystemExit(1)
selected["claimed"] = True
selected["claim_ts"] = int(time.time())
state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
candidate_path.parent.mkdir(parents=True, exist_ok=True)
with candidate_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=header)
    writer.writeheader()
    writer.writerow(
        {
            "queue": selected.get("queue", ""),
            "planned": selected.get("planned", 0),
            "running": selected.get("running", 0),
            "stalled": selected.get("stalled", 0),
            "failed": selected.get("failed", 0),
            "completed": selected.get("completed", 0),
            "total": selected.get("total", 0),
            "urgency": f"{float(selected.get('urgency', 0.0)):.3f}",
            "mtime": selected.get("mtime", 0),
            "promotion_potential": selected.get("promotion_potential", "POSSIBLE"),
            "gate_status": selected.get("gate_status", "OPEN"),
            "gate_reason": selected.get("gate_reason", ""),
            "pre_rank_score": f"{float(selected.get('pre_rank_score', 0.0)):.3f}",
            "strict_gate_status": selected.get("strict_gate_status", ""),
            "strict_gate_reason": selected.get("strict_gate_reason", ""),
            "effective_planned_count": selected.get("effective_planned_count", 0),
            "stalled_share": f"{float(selected.get('stalled_share', 0.0)):.6f}",
            "queue_yield_score": f"{float(selected.get('queue_yield_score', 0.0)):.3f}",
            "recent_yield": f"{float(selected.get('recent_yield', 0.0)):.3f}",
        }
    )
print(str(selected.get("queue") or "").strip())
PY
    rc=$?
  elif (( rc == 24 )); then
    log "ready_buffer_stale_state action=refresh exclude=${exclude_queue:-none}"
    ready_buffer_refresh "$exclude_queue" || true
    return 1
  fi
  return "$rc"
}

ready_buffer_release_claim() {
  local queue_rel="$1"
  python3 - "$READY_BUFFER_STATE_FILE" "$queue_rel" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
queue = str(sys.argv[2] or "").strip()
if not path.exists() or not queue:
    raise SystemExit(0)
try:
    data = json.loads(path.read_text(encoding='utf-8'))
except Exception:
    raise SystemExit(0)
entries = data.get("entries", []) if isinstance(data, dict) else []
if not isinstance(entries, list):
    raise SystemExit(0)
changed = False
for entry in entries:
    if not isinstance(entry, dict):
        continue
    if str(entry.get("queue") or "").strip() != queue:
        continue
    entry["claimed"] = False
    entry["claim_ts"] = 0
    changed = True
if changed:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
PY
}

process_slo_idle_with_executable_pending() {
  python3 - "$STATE_DIR/process_slo_state.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("0")
    raise SystemExit(0)
try:
    data = json.loads(path.read_text(encoding='utf-8'))
except Exception:
    print("0")
    raise SystemExit(0)
queue = data.get("queue", {}) if isinstance(data, dict) else {}
kpi = data.get("kpi", {}) if isinstance(data, dict) else {}
idle = bool(queue.get("idle_with_executable_pending")) or bool(kpi.get("idle_with_executable_pending"))
print("1" if idle else "0")
PY
}

runtime_observability_record_event() {
  local event_name="$1"
  local now_epoch="${2:-$(date +%s)}"
  python3 - "$RUNTIME_OBSERVABILITY_STATE_FILE" "$event_name" "$now_epoch" "$RUNTIME_OBSERVABILITY_WINDOW_SEC" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
event_name = str(sys.argv[2] or "").strip()
try:
    now_epoch = int(float(sys.argv[3] or 0))
except Exception:
    now_epoch = 0
try:
    window_sec = max(60, int(float(sys.argv[4] or 1800)))
except Exception:
    window_sec = 1800
if not event_name or now_epoch <= 0:
    raise SystemExit(0)

payload = {}
if path.exists():
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
if not isinstance(payload, dict):
    payload = {}
events = payload.get("events", {})
if not isinstance(events, dict):
    events = {}
series = [int(float(item or 0)) for item in list(events.get(event_name, []) or []) if str(item or "").strip()]
cutoff = now_epoch - window_sec
series = [ts for ts in series if ts >= cutoff]
series.append(now_epoch)
events[event_name] = series[-2048:]
payload["events"] = events
payload["ts_epoch"] = now_epoch
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

refresh_runtime_observability_metrics() {
  local queue_jobs child_count cpu_busy busy_now
  local vps_duty_cycle_30m metrics_missing_abort_count_30m winner_proximate_dispatch_count_30m
  local fastlane_replay_pending winner_parent_duplication_rate hot_standby_candidate_count
  queue_jobs="$(remote_active_queue_jobs)"
  child_count="$(remote_child_process_count)"
  cpu_busy="$(remote_cpu_busy_without_queue_job)"
  [[ "$queue_jobs" =~ ^[0-9]+$ ]] || queue_jobs=0
  [[ "$child_count" =~ ^[0-9]+$ ]] || child_count=0
  [[ "$cpu_busy" =~ ^[0-9]+$ ]] || cpu_busy=0
  busy_now=0
  if (( queue_jobs > 0 || cpu_busy == 1 || child_count > 0 )); then
    busy_now=1
  fi

  read -r vps_duty_cycle_30m metrics_missing_abort_count_30m winner_proximate_dispatch_count_30m fastlane_replay_pending winner_parent_duplication_rate hot_standby_candidate_count < <(
    python3 - "$RUNTIME_OBSERVABILITY_STATE_FILE" "$FULLSPAN_DECISION_STATE_FILE" "$READY_BUFFER_STATE_FILE" "$READY_BUFFER_POOL_FILE" "$busy_now" "$RUNTIME_OBSERVABILITY_WINDOW_SEC" <<'PY'
import csv
import json
import sys
import time
from pathlib import Path

state_path = Path(sys.argv[1])
fullspan_path = Path(sys.argv[2])
ready_state_path = Path(sys.argv[3])
ready_pool_path = Path(sys.argv[4])
try:
    busy_now = 1 if int(float(sys.argv[5] or 0)) > 0 else 0
except Exception:
    busy_now = 0
try:
    window_sec = max(60, int(float(sys.argv[6] or 1800)))
except Exception:
    window_sec = 1800
now_epoch = int(time.time())
cutoff = now_epoch - window_sec

payload = {}
if state_path.exists():
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
if not isinstance(payload, dict):
    payload = {}
events = payload.get("events", {})
if not isinstance(events, dict):
    events = {}
samples = payload.get("vps_busy_samples", [])
if not isinstance(samples, list):
    samples = []
fresh_samples = []
for sample in samples:
    if not isinstance(sample, dict):
        continue
    try:
        ts = int(float(sample.get("ts") or 0))
    except Exception:
        ts = 0
    if ts < cutoff:
        continue
    fresh_samples.append({"ts": ts, "busy": 1 if int(float(sample.get("busy") or 0)) > 0 else 0})
fresh_samples.append({"ts": now_epoch, "busy": busy_now})
payload["vps_busy_samples"] = fresh_samples[-4096:]

def event_count(name: str) -> int:
    series = []
    for item in list(events.get(name, []) or []):
        try:
            ts = int(float(item or 0))
        except Exception:
            ts = 0
        if ts >= cutoff:
            series.append(ts)
    events[name] = series[-2048:]
    return len(series)

fullspan = {}
if fullspan_path.exists():
    try:
        fullspan = json.loads(fullspan_path.read_text(encoding="utf-8"))
    except Exception:
        fullspan = {}
queues = fullspan.get("queues", {}) if isinstance(fullspan, dict) else {}
if not isinstance(queues, dict):
    queues = {}

fastlane_pending = 0
winner_tokens = []
seen_tokens = set()
for queue_rel, entry in queues.items():
    if not isinstance(entry, dict):
        continue
    verdict = str(entry.get("promotion_verdict") or "").strip().upper()
    try:
        strict_pass = int(float(entry.get("strict_pass_count", 0) or 0))
    except Exception:
        strict_pass = 0
    try:
        strict_groups = int(float(entry.get("strict_run_group_count", 0) or 0))
    except Exception:
        strict_groups = 0
    try:
        confirm_count = int(float(entry.get("confirm_count", 0) or 0))
    except Exception:
        confirm_count = 0
    token = str(entry.get("top_run_group") or Path(str(queue_rel)).parent.name).strip()
    if token and (strict_pass > 0 or verdict in {"PROMOTE_PENDING_CONFIRM", "PROMOTE_DEFER_CONFIRM", "PROMOTE_ELIGIBLE"}):
        if token not in seen_tokens:
            seen_tokens.add(token)
            winner_tokens.append(token)
    if verdict in {"PROMOTE_PENDING_CONFIRM", "PROMOTE_DEFER_CONFIRM"} and strict_pass > 0 and strict_groups >= 2 and confirm_count < 2:
        fastlane_pending += 1

ready_payload = {}
if ready_state_path.exists():
    try:
        ready_payload = json.loads(ready_state_path.read_text(encoding="utf-8"))
    except Exception:
        ready_payload = {}
entries = ready_payload.get("entries", []) if isinstance(ready_payload, dict) else []
if not isinstance(entries, list):
    entries = []
if not entries and ready_pool_path.exists():
    try:
        entries = list(csv.DictReader(ready_pool_path.open(newline="", encoding="utf-8")))
    except Exception:
        entries = []

parents = []
hot_standby_candidates = 0
winner_token_set = set(token for token in winner_tokens if token)
for entry in entries:
    if not isinstance(entry, dict):
        continue
    queue_rel = str(entry.get("queue") or "").strip()
    if not queue_rel:
        continue
    state_entry = queues.get(queue_rel, {})
    parent = str((state_entry or {}).get("top_run_group") or Path(queue_rel).parent.name).strip()
    if parent:
        parents.append(parent)
        if parent in winner_token_set:
            hot_standby_candidates += 1

unique_parents = len(set(parents))
dup_rate = 0.0
if parents:
    dup_rate = max(0.0, float(len(parents) - unique_parents) / float(len(parents)))

duty = 0.0
if payload["vps_busy_samples"]:
    duty = float(sum(int(sample.get("busy", 0)) for sample in payload["vps_busy_samples"])) / float(len(payload["vps_busy_samples"]))

payload["events"] = events
payload["ts_epoch"] = now_epoch
state_path.parent.mkdir(parents=True, exist_ok=True)
state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(
    "{duty:.6f} {metrics_missing_abort} {winner_dispatch} {fastlane_pending} {dup_rate:.6f} {hot_ready}".format(
        duty=duty,
        metrics_missing_abort=event_count("metrics_missing_abort"),
        winner_dispatch=event_count("winner_proximate_dispatch"),
        fastlane_pending=fastlane_pending,
        dup_rate=dup_rate,
        hot_ready=hot_standby_candidates,
    )
)
PY
  )

  fullspan_state_metric_set "vps_duty_cycle_30m" "${vps_duty_cycle_30m:-0}"
  fullspan_state_metric_set "metrics_missing_abort_count_30m" "${metrics_missing_abort_count_30m:-0}"
  fullspan_state_metric_set "winner_proximate_dispatch_count_30m" "${winner_proximate_dispatch_count_30m:-0}"
  fullspan_state_metric_set "fastlane_replay_pending" "${fastlane_replay_pending:-0}"
  fullspan_state_metric_set "winner_parent_duplication_rate" "${winner_parent_duplication_rate:-0}"
  fullspan_state_metric_set "hot_standby_candidate_count" "${hot_standby_candidate_count:-0}"
}

queue_is_winner_proximate() {
  local queue_rel="$1"
  python3 - "$FULLSPAN_DECISION_STATE_FILE" "$queue_rel" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
queue_rel = str(sys.argv[2] or "").strip()
if not path.exists() or not queue_rel:
    print("0")
    raise SystemExit(0)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("0")
    raise SystemExit(0)
queues = payload.get("queues", {}) if isinstance(payload, dict) else {}
if not isinstance(queues, dict):
    print("0")
    raise SystemExit(0)
tokens = set()
for _, entry in queues.items():
    if not isinstance(entry, dict):
        continue
    verdict = str(entry.get("promotion_verdict") or "").strip().upper()
    try:
        strict_pass = int(float(entry.get("strict_pass_count", 0) or 0))
    except Exception:
        strict_pass = 0
    if strict_pass <= 0 and verdict not in {"PROMOTE_PENDING_CONFIRM", "PROMOTE_DEFER_CONFIRM", "PROMOTE_ELIGIBLE"}:
        continue
    token = str(entry.get("top_run_group") or "").strip()
    if token:
        tokens.add(token)
queue_group = Path(queue_rel).parent.name
state_entry = queues.get(queue_rel, {})
queue_token = str((state_entry or {}).get("top_run_group") or queue_group).strip()
print("1" if queue_token and queue_token in tokens else "0")
PY
}

hot_standby_needed() {
  if ! is_truthy "$VPS_HOT_STANDBY_ENABLE"; then
    echo 0
    return 0
  fi
  local fastlane_pending ready_depth idle_pending
  fastlane_pending="$(fullspan_state_metric_get fastlane_replay_pending 0)"
  ready_depth="$(ready_buffer_depth)"
  idle_pending="$(process_slo_idle_with_executable_pending)"
  [[ "$fastlane_pending" =~ ^[0-9]+$ ]] || fastlane_pending=0
  [[ "$ready_depth" =~ ^[0-9]+$ ]] || ready_depth=0
  [[ "$idle_pending" =~ ^[0-9]+$ ]] || idle_pending=0
  if (( fastlane_pending > 0 || (ready_depth > 0 && idle_pending == 1) )); then
    echo 1
  else
    echo 0
  fi
}

maybe_prepare_hot_standby() {
  local standby_required pending_total
  standby_required="$(hot_standby_needed)"
  [[ "$standby_required" =~ ^[0-9]+$ ]] || standby_required=0
  fullspan_state_metric_set "hot_standby_active" "$standby_required"
  if (( standby_required == 0 )); then
    return 0
  fi
  pending_total="$(global_pending_count)"
  [[ "$pending_total" =~ ^[0-9]+$ ]] || pending_total=0
  if (( pending_total <= 0 )); then
    return 0
  fi
  if ! vps_is_reachable; then
    if ensure_vps_ready "hot_standby"; then
      log "hot_standby_ready pending_total=$pending_total"
    else
      log "hot_standby_deferred pending_total=$pending_total"
    fi
  fi
}


_is_match_running() {
  local pattern="$1"
  local cnt
  cnt="$(_count_match_running "$pattern")"
  [[ "$cnt" =~ ^[0-9]+$ ]] || cnt=0
  (( cnt > 0 ))
}

_count_match_running() {
  local pattern="$1"
  local found
  found="$(python3 - "$pattern" "$$" "${BASHPID:-$$}" <<'PY'
import os
import sys

pattern = sys.argv[1]
self_pids = {sys.argv[2], sys.argv[3], str(os.getpid()), str(os.getppid())}
count = 0

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
    if pattern in cmd:
        count += 1

print(count)
PY
)"
  if [[ -z "$found" || ! "$found" =~ ^[0-9]+$ ]]; then
    found=0
  fi
  echo "$found"
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

capacity_search_parallel_bounds() {
  python3 - "$CAPACITY_CONTROLLER_STATE_FILE" "$CONFIRM_SLA_ESCALATION_STATE_FILE" <<'PY'
import json
import time
import sys
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def to_int(value, default):
    try:
        return int(float(value))
    except Exception:
        return int(default)


capacity = load_json(Path(sys.argv[1]))
sla_state = load_json(Path(sys.argv[2]))
policy = capacity.get("policy", {}) if isinstance(capacity.get("policy"), dict) else {}

search_min = to_int(policy.get("search_parallel_min", 2), 2)
search_max = to_int(policy.get("search_parallel_max", 16), 16)

if isinstance(sla_state, dict) and bool(sla_state.get("active")):
    until_epoch = to_int(sla_state.get("until_epoch", 0), 0)
    now_epoch = int(time.time())
    if until_epoch <= 0 or now_epoch <= until_epoch:
        sla_policy = sla_state.get("policy", {}) if isinstance(sla_state.get("policy"), dict) else {}
        if "search_parallel_max" in sla_policy:
            search_max = min(search_max, to_int(sla_policy.get("search_parallel_max"), search_max))
        if "search_parallel_min" in sla_policy:
            search_min = max(search_min, to_int(sla_policy.get("search_parallel_min"), search_min))

if search_max < search_min:
    search_max = search_min

print(f"{search_min},{search_max}")
PY
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

  if (( pending >= 320 || total >= 400 )); then
    p=40
  elif (( pending >= 160 || total >= 220 )); then
    p=32
  elif (( pending >= 80 || total >= 140 )); then
    p=24
  elif (( pending >= 40 || total >= 80 )); then
    p=20
  elif (( total <= 24 )); then
    if awk -v v="$load1" 'BEGIN { exit (v >= 14.0) ? 0 : 1 }'; then
      p=12
    else
      p=24
    fi
  elif (( total <= 60 )); then
    p=18
  else
    p=14
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
  if (( running > 0 || stalled > 0 )); then
    if awk -v v="$rate" -v th="$ADAPTIVE_LOW_RATE_THRESHOLD" 'BEGIN { exit (v+0 <= th+0) ? 0 : 1 }'; then
      p=$((p - 1))
    elif awk -v v="$rate" -v th="$ADAPTIVE_HIGH_RATE_THRESHOLD" 'BEGIN { exit (v+0 >= th+0) ? 0 : 1 }'; then
      p=$((p + 1))
    fi
  fi

  # Load-aware boost in search mode to avoid underutilizing free VPS headroom.
  if [[ "$cause" == "UNKNOWN" || "$cause" == "DATA" ]]; then
    if awk -v v="$load1" 'BEGIN { exit (v+0 <= 2.0) ? 0 : 1 }'; then
      p=$((p + 8))
    elif awk -v v="$load1" 'BEGIN { exit (v+0 <= 4.0) ? 0 : 1 }'; then
      p=$((p + 4))
    elif awk -v v="$load1" 'BEGIN { exit (v+0 <= 6.0) ? 0 : 1 }'; then
      p=$((p + 2))
    elif awk -v v="$load1" 'BEGIN { exit (v+0 >= 12.0) ? 0 : 1 }'; then
      p=$((p - 4))
    fi
  fi

  if [[ ! "$SEARCH_PARALLEL_ABS_MAX" =~ ^[0-9]+$ ]]; then
    SEARCH_PARALLEL_ABS_MAX=48
  fi
  if (( SEARCH_PARALLEL_ABS_MAX < 2 )); then
    SEARCH_PARALLEL_ABS_MAX=2
  fi
  if (( p > SEARCH_PARALLEL_ABS_MAX )); then
    p=$SEARCH_PARALLEL_ABS_MAX
  fi
  if (( p < 2 )); then
    p=2
  fi

  local cap_min cap_max
  IFS=',' read -r cap_min cap_max < <(capacity_search_parallel_bounds)
  if [[ ! "$cap_min" =~ ^[0-9]+$ ]]; then
    cap_min=2
  fi
  if [[ ! "$cap_max" =~ ^[0-9]+$ ]]; then
    cap_max=12
  fi
  if (( cap_max < cap_min )); then
    cap_max=$cap_min
  fi
  if (( p < cap_min )); then
    p=$cap_min
  fi
  if (( p > cap_max )); then
    p=$cap_max
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
  local selection_policy="${FULLSPAN_POLICY_NAME:-}"
  local selection_mode="${PROMOTION_SELECTION_MODE:-}"
  local selection_profile="${PROMOTION_SELECTION_PROFILE:-}"
  local note_promotion_verdict="${promotion_verdict:-}"
  local note_gate_status="${gate_status:-}"
  local note_gate_reason="${gate_reason:-}"
  local note_pre_rank_score="${pre_rank_score:-}"
  local ranking_primary_key="${RANKING_PRIMARY_KEY:-score_fullspan_v1}"

  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  python3 - "$DECISION_NOTES_FILE" "$queue" "$action" "$reason" "$next_step" "$ts" "$selection_policy" "$selection_mode" "$selection_profile" "$note_promotion_verdict" "$note_gate_status" "$note_gate_reason" "$note_pre_rank_score" "$ranking_primary_key" <<'PY'
import json
import hashlib
import re
import sys
from pathlib import Path

args = sys.argv[1:]
path, queue, action, reason, next_step, ts = args[:6]
selection_policy = args[6] if len(args) > 6 else ""
selection_mode = args[7] if len(args) > 7 else ""
selection_profile = args[8] if len(args) > 8 else ""
promotion_verdict = args[9] if len(args) > 9 else ""
gate_status = args[10] if len(args) > 10 else ""
gate_reason = args[11] if len(args) > 11 else ""
pre_rank_score = args[12] if len(args) > 12 else ""
ranking_primary_key = args[13] if len(args) > 13 else ""
p = Path(path)
p.parent.mkdir(parents=True, exist_ok=True)
rec = {
    "ts": ts,
    "queue": queue,
    "action": action,
    "reason": reason,
    "next_step": next_step,
    "selection_policy": selection_policy,
    "selection_mode": selection_mode,
    "selection_profile": selection_profile,
    "promotion_verdict": promotion_verdict,
    "gate_status": gate_status,
    "gate_reason": gate_reason,
    "pre_rank_score": pre_rank_score,
    "ranking_primary_key": ranking_primary_key,
}
with p.open("a", encoding="utf-8") as f:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
PY
}

surrogate_gate_decision() {
  local queue_rel="$1"
  local queue_abs="$ROOT_DIR/$queue_rel"
  python3 - "$GATE_SURROGATE_STATE_FILE" "$queue_rel" "$queue_abs" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
queue_rel = str(sys.argv[2] or "").strip()
queue_abs = str(sys.argv[3] or "").strip()
queue_rooted = "/" + queue_rel.lstrip("/") if queue_rel else ""

keys = {queue_rel, queue_abs, queue_rooted}
keys.discard("")

decision = "allow"
reason = ""

if not state_path.exists():
    print(f"{decision}\t{reason}")
    raise SystemExit(0)

try:
    data = json.loads(state_path.read_text(encoding="utf-8"))
except Exception:
    print(f"{decision}\tinvalid_surrogate_state_json")
    raise SystemExit(0)

if not isinstance(data, dict):
    print(f"{decision}\tinvalid_surrogate_state_payload")
    raise SystemExit(0)

def parse_entry(entry):
    if isinstance(entry, str):
        return str(entry).strip().lower(), ""
    if isinstance(entry, dict):
        d = str(entry.get("decision") or entry.get("status") or "").strip().lower()
        r = str(entry.get("reason") or entry.get("note") or entry.get("message") or "").strip()
        return d, r
    return "", ""

selected = None
for bucket_name in ("queues", "queue_decisions", "by_queue"):
    bucket = data.get(bucket_name)
    if isinstance(bucket, dict):
        for key in keys:
            if key in bucket:
                selected = bucket[key]
                break
    if selected is not None:
        break
    if isinstance(bucket, list):
        for item in bucket:
            if not isinstance(item, dict):
                continue
            q = str(item.get("queue") or item.get("queue_rel") or item.get("queue_path") or "").strip()
            if q in keys:
                selected = item
                break
    if selected is not None:
        break

if selected is None:
    selected = data

parsed_decision, parsed_reason = parse_entry(selected)
if parsed_decision in {"allow", "reject", "refine"}:
    decision = parsed_decision
if parsed_reason:
    reason = parsed_reason
print(f"{decision}\t{reason}")
PY
}

surrogate_gate_state_needs_refresh() {
  local queue_rel="$1"
  local queue_abs="${2:-$ROOT_DIR/$queue_rel}"
  python3 - "$GATE_SURROGATE_STATE_FILE" "$queue_rel" "$queue_abs" <<'PY'
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

state_path = Path(sys.argv[1])
queue_rel = str(sys.argv[2] or "").strip()
queue_abs = Path(str(sys.argv[3] or "").strip())
fresh_sec = 90

if not state_path.exists():
    print("1")
    raise SystemExit(0)

try:
    data = json.loads(state_path.read_text(encoding="utf-8"))
except Exception:
    print("1")
    raise SystemExit(0)

if not isinstance(data, dict):
    print("1")
    raise SystemExit(0)

ts = str(data.get("ts") or "").strip()
age_sec = fresh_sec + 1
if ts:
    try:
        parsed = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        age_sec = max(0, int(time.time() - int(parsed.timestamp())))
    except Exception:
        age_sec = fresh_sec + 1

queues = data.get("queues")
has_queue = isinstance(queues, dict) and queue_rel in queues

state_mtime = 0.0
queue_mtime = 0.0
try:
    state_mtime = float(state_path.stat().st_mtime)
except Exception:
    state_mtime = 0.0
try:
    if queue_abs.exists():
        queue_mtime = float(queue_abs.stat().st_mtime)
except Exception:
    queue_mtime = 0.0

refresh_needed = age_sec > fresh_sec or not has_queue
if not refresh_needed and queue_mtime > state_mtime:
    refresh_needed = True

print("1" if refresh_needed else "0")
PY
}

refresh_gate_surrogate_state() {
  local queue_rel="$1"
  local refresh_needed="1"
  refresh_needed="$(surrogate_gate_state_needs_refresh "$queue_rel" "$ROOT_DIR/$queue_rel" 2>/dev/null || printf '1')"
  if [[ "$refresh_needed" != "1" ]]; then
    return 0
  fi
  local py_bin="$ROOT_DIR/.venv/bin/python"
  [[ -x "$py_bin" ]] || py_bin="$(command -v python3)"
  if [[ -z "$py_bin" ]]; then
    log "surrogate_gate_refresh queue=$queue_rel status=skip reason=no_python"
    return 1
  fi
  if "$py_bin" "$ROOT_DIR/scripts/optimization/gate_surrogate_agent.py" --root "$ROOT_DIR" >/dev/null 2>&1; then
    "$py_bin" "$ROOT_DIR/scripts/optimization/search_director_agent.py" --root "$ROOT_DIR" >/dev/null 2>&1 || true
    log "surrogate_gate_refresh queue=$queue_rel status=ok"
    return 0
  fi
  log "surrogate_gate_refresh queue=$queue_rel status=failed"
  return 1
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
import hashlib
import re
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

evo_re = re.compile(r"\b(evo_[0-9a-f]{8,64})\b", re.IGNORECASE)

def derive_candidate_uid(top_run_group: str, top_variant: str, top_score: str) -> str:
    for token in (top_variant, top_run_group):
        m = evo_re.search(str(token or ""))
        if m:
            return m.group(1).lower()
    parts = [
        str(top_run_group or "").strip().lower(),
        str(top_variant or "").strip().lower(),
        str(top_score or "").strip(),
    ]
    if not any(parts):
        return ""
    return "cand_" + hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]

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

prev_entry = queues.get(queue, {})
if not isinstance(prev_entry, dict):
    prev_entry = {}

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
    "candidate_uid": str(prev_entry.get("candidate_uid") or derive_candidate_uid(top_run_group, top_variant, top_score)),
    "cutover_permission": "FAIL_CLOSED",
    "cutover_ready": False,
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
    if i + 1 >= len(sys.argv):
        break
    key = sys.argv[i]
    val = sys.argv[i + 1]
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

vps_runtime_load() {
  local loaded
  if [[ -f "$VPS_RECOVERY_STATE_FILE" ]]; then
    loaded="$(python3 - "$VPS_RECOVERY_STATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    obj = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    obj = {}

def to_int(value):
    try:
        return int(float(value or 0))
    except Exception:
        return 0

print(
    "{},{},{},{},{}".format(
        to_int(obj.get("fail_streak")),
        to_int(obj.get("unreachable_since_epoch")),
        to_int(obj.get("last_recover_epoch")),
        to_int(obj.get("next_retry_epoch")),
        to_int(obj.get("last_force_cycle_epoch")),
    )
)
PY
)"
    IFS=',' read -r vps_runtime_fail_streak vps_runtime_unreachable_since vps_runtime_last_recover_epoch vps_runtime_next_retry_epoch vps_runtime_last_force_cycle_epoch <<< "$loaded"
  else
    vps_runtime_fail_streak="$(fullspan_state_metric_get vps_recover_consecutive_fail 0)"
    vps_runtime_unreachable_since="$(fullspan_state_metric_get vps_unreachable_since_epoch 0)"
    vps_runtime_last_recover_epoch="$(fullspan_state_metric_get vps_recover_last_epoch 0)"
    vps_runtime_next_retry_epoch="$(fullspan_state_metric_get vps_recover_next_attempt_epoch 0)"
    vps_runtime_last_force_cycle_epoch="$(fullspan_state_metric_get vps_force_cycle_last_epoch 0)"
  fi

  [[ "$vps_runtime_fail_streak" =~ ^[0-9]+$ ]] || vps_runtime_fail_streak=0
  [[ "$vps_runtime_unreachable_since" =~ ^[0-9]+$ ]] || vps_runtime_unreachable_since=0
  [[ "$vps_runtime_last_recover_epoch" =~ ^[0-9]+$ ]] || vps_runtime_last_recover_epoch=0
  [[ "$vps_runtime_next_retry_epoch" =~ ^[0-9]+$ ]] || vps_runtime_next_retry_epoch=0
  [[ "$vps_runtime_last_force_cycle_epoch" =~ ^[0-9]+$ ]] || vps_runtime_last_force_cycle_epoch=0
}

vps_runtime_save() {
  python3 - "$VPS_RECOVERY_STATE_FILE" "$vps_runtime_fail_streak" "$vps_runtime_unreachable_since" "$vps_runtime_last_recover_epoch" "$vps_runtime_next_retry_epoch" "$vps_runtime_last_force_cycle_epoch" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])

def to_int(value):
    try:
        return int(float(value or 0))
    except Exception:
        return 0

payload = {
    "version": 1,
    "fail_streak": to_int(sys.argv[2]),
    "unreachable_since_epoch": to_int(sys.argv[3]),
    "last_recover_epoch": to_int(sys.argv[4]),
    "next_retry_epoch": to_int(sys.argv[5]),
    "last_force_cycle_epoch": to_int(sys.argv[6]),
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

vps_runtime_sync_metrics() {
  fullspan_state_metric_set "vps_recover_consecutive_fail" "$vps_runtime_fail_streak"
  fullspan_state_metric_set "vps_unreachable_since_epoch" "$vps_runtime_unreachable_since"
  fullspan_state_metric_set "vps_recover_last_epoch" "$vps_runtime_last_recover_epoch"
  fullspan_state_metric_set "vps_recover_next_attempt_epoch" "$vps_runtime_next_retry_epoch"
  fullspan_state_metric_set "vps_force_cycle_last_epoch" "$vps_runtime_last_force_cycle_epoch"
}

vps_runtime_commit() {
  vps_runtime_save
  vps_runtime_sync_metrics
}

vps_runtime_next_retry_get() {
  if [[ "$vps_runtime_next_retry_epoch" =~ ^[0-9]+$ ]]; then
    echo "$vps_runtime_next_retry_epoch"
    return 0
  fi
  fullspan_state_metric_get vps_recover_next_attempt_epoch 0
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


derive_candidate_uid() {
  local top_run_group="${1:-}"
  local top_variant="${2:-}"
  local top_score="${3:-}"
  if [[ ! -f "$ROOT_DIR/scripts/optimization/fullspan_lineage.py" ]]; then
    echo ""
    return 0
  fi
  ./.venv/bin/python scripts/optimization/fullspan_lineage.py derive \
    --top-run-group "$top_run_group" \
    --top-variant "$top_variant" \
    --top-score "$top_score" 2>/dev/null || true
}

fullspan_confirm_count_for_queue() {
  local candidate_uid="$1"
  if [[ -z "$candidate_uid" ]]; then
    echo 0
    return 0
  fi
  if [[ ! -f "$ROOT_DIR/scripts/optimization/fullspan_lineage.py" ]]; then
    echo 0
    return 0
  fi
  ./.venv/bin/python scripts/optimization/fullspan_lineage.py count \
    --registry "$CONFIRM_LINEAGE_REGISTRY_FILE" \
    --run-index "$RUN_INDEX_PATH" \
    --candidate-uid "$candidate_uid" \
    --value count 2>/dev/null || echo 0
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
  local candidate_uid

  state_verdict="$(fullspan_state_get "$queue_rel" "promotion_verdict" "ANALYZE")"
  if [[ "$state_verdict" != "PROMOTE_PENDING_CONFIRM" && "$state_verdict" != "PROMOTE_DEFER_CONFIRM" && "$state_verdict" != "PROMOTE_ELIGIBLE" ]]; then
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
  candidate_uid="$(fullspan_state_get "$queue_rel" "candidate_uid" "")"

  if [[ "$strict_run_group_count" -lt "$FULLSPAN_CONFIRM_MIN_GROUPS" ]]; then
    return 0
  fi

  if [[ -z "$candidate_uid" ]]; then
    candidate_uid="$(derive_candidate_uid "$top_run_group" "$top_variant" "$top_score")"
  fi

  if [[ -z "$candidate_uid" ]]; then
    return 0
  fi

  local live_confirm_count
  live_confirm_count="$(fullspan_confirm_count_for_queue "$candidate_uid")"

  local run_groups_csv
  run_groups_csv="$(fullspan_state_run_groups_csv "$queue_rel")"

  if (( live_confirm_count != state_confirm_count )); then
    local next_verdict="$state_verdict"
    local next_reason="$rejection_reason"
    if (( strict_run_group_count >= FULLSPAN_CONFIRM_MIN_GROUPS )); then
      next_verdict="PROMOTE_PENDING_CONFIRM"
      if (( live_confirm_count >= FULLSPAN_CONFIRM_MIN_REPLIES )); then
        next_reason="confirm_ready_pending_gatekeeper"
      else
        next_reason="pending_confirm"
      fi
    fi
    fullspan_state_set "$queue_rel" "$next_verdict" \
      "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
      "$next_reason" "$live_confirm_count" "$strict_gate_status" "$strict_gate_reason" "$run_groups_csv" "$strict_summary_path"
    fullspan_state_queue_set "$queue_rel" "candidate_uid" "$candidate_uid"
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
      local candidate_uid
      confirm_count="0"
      candidate_uid=""
      if [[ -n "$top_run_group" || -n "$top_variant" ]]; then
        candidate_uid="$(derive_candidate_uid "$top_run_group" "$top_variant" "$top_score")"
      fi
      if [[ "$strict_pass_count" -gt 0 ]]; then
        confirm_count="$(fullspan_confirm_count_for_queue "$candidate_uid")"
      fi

      if [[ "$strict_pass_count" -le 0 ]]; then
        fullspan_state_set "$queue_rel" "REJECT" \
          "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
          "${strict_rejection_reason:-METRICS_MISSING}" "$confirm_count" "FULLSPAN_PREFILTER_REJECT" "" "$strict_run_groups" "$cycle_summary"
        [[ -n "$candidate_uid" ]] && fullspan_state_queue_set "$queue_rel" "candidate_uid" "$candidate_uid"
      elif (( strict_run_group_count >= FULLSPAN_CONFIRM_MIN_GROUPS )); then
        if (( confirm_count >= FULLSPAN_CONFIRM_MIN_REPLIES )); then
          fullspan_state_set "$queue_rel" "PROMOTE_PENDING_CONFIRM" \
            "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
            "confirm_ready_pending_gatekeeper" "$confirm_count" "FULLSPAN_PREFILTER_PASSED" "" "$strict_run_groups" "$cycle_summary"
        else
          fullspan_state_set "$queue_rel" "PROMOTE_PENDING_CONFIRM" \
            "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
            "pending_confirm" "$confirm_count" "FULLSPAN_PREFILTER_PASSED" "" "$strict_run_groups" "$cycle_summary"
        fi
        [[ -n "$candidate_uid" ]] && fullspan_state_queue_set "$queue_rel" "candidate_uid" "$candidate_uid"
      else
        fullspan_state_set "$queue_rel" "PROMOTE_DEFER_CONFIRM" \
          "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
          "insufficient_run_groups" "$confirm_count" "FULLSPAN_PREFILTER_PASSED" "" "$strict_run_groups" "$cycle_summary"
        [[ -n "$candidate_uid" ]] && fullspan_state_queue_set "$queue_rel" "candidate_uid" "$candidate_uid"
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
  local candidate_uid
  confirm_count="0"
  candidate_uid=""
  if [[ -n "$top_run_group" || -n "$top_variant" ]]; then
    candidate_uid="$(derive_candidate_uid "$top_run_group" "$top_variant" "$top_score")"
  fi
  if [[ "$strict_pass_count" -gt 0 ]]; then
    confirm_count="$(fullspan_confirm_count_for_queue "$candidate_uid")"
  fi

  if (( cycle_rc == 0 )); then
    if (( strict_pass_count <= 0 )); then
      fullspan_state_set "$queue_rel" "REJECT" \
        "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
        "${strict_rejection_reason:-METRICS_MISSING}" "$confirm_count" "FULLSPAN_PREFILTER_REJECT" "" "$strict_run_groups" "$cycle_summary"
      [[ -n "$candidate_uid" ]] && fullspan_state_queue_set "$queue_rel" "candidate_uid" "$candidate_uid"
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
          fullspan_state_set "$queue_rel" "PROMOTE_PENDING_CONFIRM" \
            "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
            "confirm_ready_pending_gatekeeper" "$confirm_count" "FULLSPAN_PREFILTER_PASSED" "" "$strict_run_groups" "$cycle_summary"
          log_decision_note "$queue_rel" "FULLSPAN_STRICT_CONFIRM_READY" "strict_pass_and_min_run_groups+confirm" "await_gatekeeper_eligibility"
        else
          fullspan_state_set "$queue_rel" "PROMOTE_PENDING_CONFIRM" \
            "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
            "pending_confirm" "$confirm_count" "FULLSPAN_PREFILTER_PASSED" "" "$strict_run_groups" "$cycle_summary"
          fullspan_state_metric_set "confirm_pending_count" "$(( $(fullspan_state_metric_get confirm_pending_count 0) + 1 ))"
          log_decision_note "$queue_rel" "FULLSPAN_STRICT_PENDING_CONFIRM" "strict_pass_run_groups=$strict_run_group_count confirm_count=$confirm_count" "need_vps_confirm_replay"
        fi
        [[ -n "$candidate_uid" ]] && fullspan_state_queue_set "$queue_rel" "candidate_uid" "$candidate_uid"
      else
        fullspan_state_set "$queue_rel" "PROMOTE_DEFER_CONFIRM" \
          "$strict_pass_count" "$strict_run_group_count" "$top_run_group" "$top_variant" "$top_score" \
          "insufficient_run_groups" "$confirm_count" "FULLSPAN_PREFILTER_PASSED" "" "$strict_run_groups" "$cycle_summary"
        fullspan_state_metric_set "confirm_pending_count" "$(( $(fullspan_state_metric_get confirm_pending_count 0) + 1 ))"
        log_decision_note "$queue_rel" "FULLSPAN_STRICT_PASS" "strict_pass_count=$strict_pass_count run_groups=$strict_run_group_count" "need_additional_run_groups"
        [[ -n "$candidate_uid" ]] && fullspan_state_queue_set "$queue_rel" "candidate_uid" "$candidate_uid"
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

early_stop_low_yield_queue() {
  local queue_rel="$1"
  local queue_path="$ROOT_DIR/$queue_rel"
  python3 - "$queue_rel" "$queue_path" "$RUN_INDEX_PATH" "$EARLY_STOP_MIN_COMPLETED" "$EARLY_STOP_FAIL_FRACTION" "$EARLY_STOP_DOMINANT_FRACTION" "$EARLY_STOP_DOMINANT_MIN" "$EARLY_ABORT_MIN_COMPLETED" "$EARLY_ABORT_ZERO_ACTIVITY_SHARE" "$EARLY_ABORT_ZERO_ACTIVITY_MIN" <<'PY'
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

queue_rel = str(sys.argv[1] or "").strip()
queue_path = Path(sys.argv[2])
run_index_path = Path(sys.argv[3])
min_completed = int(float(sys.argv[4] or 8))
fail_fraction_gate = float(sys.argv[5] or 0.75)
dominant_fraction_gate = float(sys.argv[6] or 0.70)
dominant_min = int(float(sys.argv[7] or 6))
zero_activity_min_completed = int(float(sys.argv[8] or 12))
zero_activity_share_gate = float(sys.argv[9] or 0.80)
zero_activity_min = int(float(sys.argv[10] or 6))

INITIAL_CAPITAL = 1000.0

payload = {
    "trigger": False,
    "reason": "",
    "changed": 0,
    "completed": 0,
    "evaluated": 0,
    "fail_fraction": 0.0,
    "dominant_fraction": 0.0,
    "zero_activity_fraction": 0.0,
}

if (
    not queue_path.exists()
    or not run_index_path.exists()
    or "confirm_fastlane_" in queue_rel
    or "/confirm_fastlane_" in queue_rel
):
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(0)

try:
    rows = list(csv.DictReader(queue_path.open(newline="", encoding="utf-8")))
except Exception:
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(0)

run_index = {}
try:
    with run_index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            run_index[str(row.get("run_id") or "").strip()] = row
except Exception:
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(0)

completed_rows = [r for r in rows if str(r.get("status") or "").strip().lower() == "completed"]
payload["completed"] = len(completed_rows)
if len(completed_rows) < max(1, min_completed):
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(0)

def to_float(value, default=None):
    try:
        return float((value or "").strip())
    except Exception:
        return default

def is_true(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}

def hard_gate_reason(row):
    if not row:
        return "METRICS_MISSING"
    if not is_true(row.get("metrics_present")):
        return "METRICS_MISSING"
    if to_float(row.get("total_trades"), 0.0) < 200:
        return "TRADES_FAIL"
    if to_float(row.get("total_pairs_traded"), 0.0) < 20:
        return "PAIRS_FAIL"
    dd = abs(to_float(row.get("max_drawdown_on_equity"), 0.0) or 0.0)
    if dd > 0.20:
        return "DD_FAIL"
    if to_float(row.get("total_pnl"), 0.0) < 0:
        return "ECONOMIC_FAIL"
    worst_step = to_float(row.get("tail_loss_worst_period_pnl"), None)
    if worst_step is None:
        worst_step = to_float(row.get("tail_loss_worst_pair_pnl"), 0.0)
    if (worst_step or 0.0) < (-0.20 * INITIAL_CAPITAL):
        return "STEP_FAIL"
    return ""

by_base = defaultdict(lambda: {"holdout": None, "stress": None})
for row in completed_rows:
    results_dir = str(row.get("results_dir") or "").strip()
    if not results_dir:
        continue
    run_id = Path(results_dir).name.strip()
    base = re.sub(r"^(holdout_|stress_)", "", run_id)
    if run_id.startswith("holdout_"):
        by_base[base]["holdout"] = run_index.get(run_id)
    elif run_id.startswith("stress_"):
        by_base[base]["stress"] = run_index.get(run_id)
    else:
        by_base[base]["holdout"] = run_index.get(run_id)

fail_counter = Counter()
pass_count = 0
eval_count = 0
zero_activity_count = 0

def is_zero_activity(row):
    if not row:
        return True
    if not is_true(row.get("metrics_present")):
        return True
    trades = to_float(row.get("total_trades"), 0.0) or 0.0
    pairs = to_float(row.get("total_pairs_traded"), 0.0) or 0.0
    pnl = to_float(row.get("total_pnl"), 0.0)
    return trades <= 0.0 or pairs <= 0.0 or pnl is None

for bundle in by_base.values():
    h = bundle.get("holdout")
    s = bundle.get("stress")
    if h is None and s is None:
        continue
    eval_count += 1
    if is_zero_activity(h) and is_zero_activity(s):
        zero_activity_count += 1
    h_reason = hard_gate_reason(h) if h is not None else "METRICS_MISSING"
    s_reason = hard_gate_reason(s) if s is not None else "METRICS_MISSING"
    if not h_reason and not s_reason:
        pass_count += 1
    else:
        fail_counter[h_reason or s_reason] += 1

payload["evaluated"] = eval_count
if eval_count < max(1, min_completed):
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(0)

zero_activity_fraction = float(zero_activity_count / eval_count) if eval_count > 0 else 0.0
payload["zero_activity_fraction"] = zero_activity_fraction
if (
    eval_count >= max(1, zero_activity_min_completed)
    and zero_activity_count >= max(1, zero_activity_min)
    and zero_activity_fraction >= zero_activity_share_gate
    and pass_count <= 0
):
    changed = 0
    for row in rows:
        status = str(row.get("status") or "").strip().lower()
        if status in {"planned", "stalled", "failed", "error"}:
            row["status"] = "skipped"
            changed += 1
    if changed > 0 and rows:
        with queue_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    payload["trigger"] = bool(changed > 0)
    payload["changed"] = changed
    payload["reason"] = "EARLY_ABORT_ZERO_ACTIVITY"
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(0)

fail_count = int(sum(fail_counter.values()))
fail_fraction = float(fail_count / eval_count) if eval_count > 0 else 0.0
payload["fail_fraction"] = fail_fraction
if fail_count <= 0 or fail_fraction < fail_fraction_gate:
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(0)

dominant_reason, dominant_count = fail_counter.most_common(1)[0]
dominant_fraction = float(dominant_count / fail_count) if fail_count > 0 else 0.0
payload["dominant_fraction"] = dominant_fraction
if dominant_count < dominant_min or dominant_fraction < dominant_fraction_gate:
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(0)

changed = 0
for row in rows:
    status = str(row.get("status") or "").strip().lower()
    if status in {"planned", "stalled", "failed", "error"}:
        row["status"] = "skipped"
        changed += 1

if changed > 0 and rows:
    with queue_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

payload["trigger"] = bool(changed > 0)
payload["changed"] = changed
payload["reason"] = f"EARLY_ABORT_LOW_INFORMATION_{dominant_reason}"
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
  remote_count="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=6 "$SERVER_USER@$SERVER_IP" "python3 - <<'PY'
import os
patterns = ('watch_wfa_queue.sh', 'run_wfa_queue.py', 'run_wfa_fullcpu.sh', 'walk_forward')
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
PY" || true)"
  if [[ -z "$remote_count" ]]; then
    echo 0
    return
  fi
  echo "$remote_count"
}

remote_active_queue_jobs() {
  local remote_count
  remote_count="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=6 "$SERVER_USER@$SERVER_IP" "python3 - <<'PY'
import os
count = 0
for pid in os.listdir('/proc'):
    if not pid.isdigit():
        continue
    try:
        cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\\x00', b' ').decode('utf-8', 'ignore').strip()
    except Exception:
        continue
    if not cmd or 'python3 - <<' in cmd or 'pgrep -f' in cmd:
        continue
    if 'run_wfa_queue.py --queue' in cmd:
        count += 1
print(count)
PY" || true)"
  [[ -n "$remote_count" && "$remote_count" =~ ^[0-9]+$ ]] || remote_count=0
  echo "$remote_count"
}

remote_child_process_count() {
  remote_runner_count
}

remote_cpu_busy_without_queue_job() {
  local child_count queue_jobs load1
  child_count="$(remote_child_process_count)"
  queue_jobs="$(remote_active_queue_jobs)"
  load1="$(get_vps_load)"
  [[ "$child_count" =~ ^[0-9]+$ ]] || child_count=0
  [[ "$queue_jobs" =~ ^[0-9]+$ ]] || queue_jobs=0
  if [[ -z "$load1" ]]; then
    load1=0
  fi
  if (( queue_jobs == 0 && child_count > 0 )) && awk -v v="$load1" 'BEGIN { exit (v + 0.0 >= 1.5) ? 0 : 1 }'; then
    echo 1
    return
  fi
  echo 0
}

refresh_remote_runtime_metrics() {
  local queue_jobs child_count cpu_busy
  queue_jobs="$(remote_active_queue_jobs)"
  child_count="$(remote_child_process_count)"
  cpu_busy="$(remote_cpu_busy_without_queue_job)"
  [[ "$queue_jobs" =~ ^[0-9]+$ ]] || queue_jobs=0
  [[ "$child_count" =~ ^[0-9]+$ ]] || child_count=0
  [[ "$cpu_busy" =~ ^[0-9]+$ ]] || cpu_busy=0
  fullspan_state_metric_set "remote_active_queue_jobs" "$queue_jobs"
  fullspan_state_metric_set "remote_queue_job_count" "$queue_jobs"
  fullspan_state_metric_set "remote_child_process_count" "$child_count"
  fullspan_state_metric_set "cpu_busy_without_queue_job" "$cpu_busy"
}

remote_queue_running() {
  local queue_rel="$1"
  local cnt
  cnt="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=6 "$SERVER_USER@$SERVER_IP" "python3 - <<'PY'
import os
needle = \"run_wfa_queue.py --queue ${queue_rel}\"
count = 0
for pid in os.listdir('/proc'):
    if not pid.isdigit():
        continue
    try:
        cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\\x00', b' ').decode('utf-8', 'ignore').strip()
    except Exception:
        continue
    if not cmd or 'python3 - <<' in cmd or 'pgrep -af' in cmd:
        continue
    if needle in cmd:
        count += 1
print(count)
PY" 2>/dev/null || echo 0)"
  [[ -n "$cnt" && "$cnt" =~ ^[0-9]+$ ]] || cnt=0
  (( cnt > 0 ))
}

local_powered_queue_running() {
  local queue_rel="$1"
  local cnt
  cnt="$(_count_match_running "run_wfa_queue_powered.py --queue ${queue_rel}")"
  [[ -n "$cnt" && "$cnt" =~ ^[0-9]+$ ]] || cnt=0
  (( cnt > 0 ))
}

sla_watch_queue() {
  local queue_rel="$1"
  local pending="$2"
  local running="$3"
  local state_verdict="$4"

  if (( pending > 0 && running == 0 )); then
    local rc remote_child_count cpu_busy_without_queue_job
    rc="$(remote_active_queue_jobs)"
    remote_child_count="$(remote_child_process_count)"
    cpu_busy_without_queue_job="$(remote_cpu_busy_without_queue_job)"
    [[ "$rc" =~ ^[0-9]+$ ]] || rc=0
    [[ "$remote_child_count" =~ ^[0-9]+$ ]] || remote_child_count=0
    [[ "$cpu_busy_without_queue_job" =~ ^[0-9]+$ ]] || cpu_busy_without_queue_job=0
    if (( rc == 0 && cpu_busy_without_queue_job == 0 )); then
      local idle_streak="${vps_idle_pending_streak_by_queue[$queue_rel]:-0}"
      idle_streak=$((idle_streak + 1))
      vps_idle_pending_streak_by_queue["$queue_rel"]="$idle_streak"
      if (( idle_streak >= SLA_VPS_IDLE_PENDING_CYCLES )); then
        fullspan_state_metric_inc "sla_vps_idle_with_pending_trigger" 1
        log_decision_note "$queue_rel" "SLA_VPS_IDLE_WITH_PENDING" "streak=$idle_streak pending=$pending remote_queue_jobs=$rc remote_child_processes=$remote_child_count" "watchdog_or_manual_intervention"
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
  busy_reason=""
  if _is_match_running "watch_wfa_queue.sh --queue"; then
    busy_reason="local_watch_wfa_queue"
    return 0
  fi
  if _is_match_running "scripts/optimization/run_wfa_queue.py --queue"; then
    busy_reason="local_run_wfa_queue"
    return 0
  fi

  local local_powered_count
  local max_local
  local_powered_count="$(_count_match_running "run_wfa_queue_powered.py --queue")"
  [[ "$local_powered_count" =~ ^[0-9]+$ ]] || local_powered_count=0
  max_local="$DRIVER_MAX_LOCAL_SEARCH_RUNNERS"
  [[ "$max_local" =~ ^[0-9]+$ ]] || max_local=3
  if (( max_local < 1 )); then
    max_local=1
  fi
  if (( local_powered_count >= max_local )); then
    busy_reason="local_powered_capacity:${local_powered_count}/${max_local}"
    return 0
  fi

  local remote_count
  local max_remote
  remote_count="$(remote_runner_count)"
  [[ "$remote_count" =~ ^[0-9]+$ ]] || remote_count=0
  max_remote="$DRIVER_MAX_REMOTE_RUNNERS"
  [[ "$max_remote" =~ ^[0-9]+$ ]] || max_remote=64
  if (( max_remote < 1 )); then
    max_remote=1
  fi
  if (( remote_count >= max_remote )); then
    busy_reason="remote_runner_capacity:${remote_count}/${max_remote}"
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

queue_hygiene_snapshot() {
  local queue_rel="$1"
  python3 - "$ROOT_DIR/$queue_rel" "$ROOT_DIR" "$RUN_INDEX_PATH" <<'PY'
import csv
import sys
from pathlib import Path

queue_path = Path(sys.argv[1])
root = Path(sys.argv[2])
run_index_path = Path(sys.argv[3])

if not queue_path.exists():
    print("0 0 0 0 0 0 0 0")
    raise SystemExit(0)

try:
    rows = list(csv.DictReader(queue_path.open(newline='', encoding='utf-8')))
except Exception:
    print("0 0 0 0 0 0 0 0")
    raise SystemExit(0)

run_index = {}
if run_index_path.exists():
    try:
        with run_index_path.open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                run_index[str(row.get('run_id') or '').strip()] = row
    except Exception:
        run_index = {}

def is_true(v):
    return str(v or '').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}

def cfg_exists(path_raw):
    s = str(path_raw or '').strip()
    if not s:
        return False
    p = Path(s)
    if not p.is_absolute():
        p = root / p
    return p.exists()

pending = 0
executable_pending = 0
completed_with_metrics = 0
planned = running = stalled = failed = completed = 0

for row in rows:
    status = str(row.get('status') or '').strip().lower()
    cfg = str(row.get('config_path') or '').strip()
    if status == 'planned':
        planned += 1
    elif status == 'running':
        running += 1
    elif status == 'stalled':
        stalled += 1
    elif status in {'failed', 'error'}:
        failed += 1
    elif status == 'completed':
        completed += 1

    if status in {'planned', 'running', 'stalled', 'failed', 'error'}:
        pending += 1
        if status == 'running' or cfg_exists(cfg):
            executable_pending += 1

    if status == 'completed':
        run_id = Path(str(row.get('results_dir') or '')).name.strip()
        idx_row = run_index.get(run_id, {})
        if is_true((idx_row or {}).get('metrics_present')):
            completed_with_metrics += 1
            continue
        run_dir = Path(str(row.get('results_dir') or '').strip())
        if run_dir and not run_dir.is_absolute():
            run_dir = root / run_dir
        if run_dir.exists() and (run_dir / 'strategy_metrics.csv').exists():
            completed_with_metrics += 1

print(
    pending,
    executable_pending,
    completed_with_metrics,
    planned,
    running,
    stalled,
    failed,
    completed,
)
PY
}

global_backlog_snapshot() {
  python3 - "$QUEUE_ROOT" "$ROOT_DIR" <<'PY'
import csv
import sys
from pathlib import Path

queue_root = Path(sys.argv[1])
root = Path(sys.argv[2])

def cfg_exists(path_raw):
    s = str(path_raw or '').strip()
    if not s:
        return False
    p = Path(s)
    if not p.is_absolute():
        p = root / p
    return p.exists()

planned_exec = 0
pending_exec = 0
no_op_queues = 0

for q in sorted(queue_root.rglob('run_queue.csv')):
    q_str = str(q)
    if '/rollup/' in q_str or '/.autonomous/' in q_str:
        continue
    try:
        rows = list(csv.DictReader(q.open(newline='', encoding='utf-8')))
    except Exception:
        continue
    queue_pending = 0
    queue_exec = 0
    for row in rows:
        st = str(row.get('status') or '').strip().lower()
        if st in {'planned', 'running', 'stalled', 'failed', 'error'}:
            queue_pending += 1
            if st == 'running' or cfg_exists(row.get('config_path')):
                queue_exec += 1
                pending_exec += 1
            if st == 'planned' and cfg_exists(row.get('config_path')):
                planned_exec += 1
    if queue_pending > 0 and queue_exec == 0:
        no_op_queues += 1

print(planned_exec, pending_exec, no_op_queues)
PY
}

global_completed_metrics_count() {
  python3 - "$RUN_INDEX_PATH" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit(0)
count = 0
try:
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            status = str(row.get('status') or '').strip().lower()
            metrics = str(row.get('metrics_present') or '').strip().lower()
            if status == 'completed' and metrics in {'1', 'true', 'yes', 'y', 'on'}:
                count += 1
except Exception:
    count = 0
print(count)
PY
}

maybe_trigger_auto_seed() {
  local reason="${1:-low_backlog}"
  local now_epoch last_seed planned_exec pending_exec no_op_queues
  local ready_depth
  local remote_count=0
  local force_seed=0
  local effective_seed_pending_threshold
  now_epoch="$(date +%s)"
  last_seed="$(fullspan_state_metric_get last_seed_trigger_epoch 0)"
  [[ "$last_seed" =~ ^[0-9]+$ ]] || last_seed=0

  read -r planned_exec pending_exec no_op_queues <<< "$(global_backlog_snapshot)"
  planned_exec="${planned_exec:-0}"
  pending_exec="${pending_exec:-0}"
  no_op_queues="${no_op_queues:-0}"
  fullspan_state_metric_set "global_planned_exec" "$planned_exec"
  fullspan_state_metric_set "global_pending_exec" "$pending_exec"
  fullspan_state_metric_set "global_no_op_queue_count" "$no_op_queues"
  ready_depth="$(ready_buffer_depth)"
  [[ "$ready_depth" =~ ^[0-9]+$ ]] || ready_depth=0
  fullspan_state_metric_set "ready_buffer_depth" "$ready_depth"

  case "$reason" in
    candidate_empty|candidate_parse_empty|candidate_empty_after_reconcile|candidate_parse_empty_after_reconcile)
      remote_count="$(remote_active_queue_jobs)"
      [[ "$remote_count" =~ ^[0-9]+$ ]] || remote_count=0
      if (( remote_count == 0 )); then
        force_seed=1
      fi
      ;;
  esac
  fullspan_state_metric_set "auto_seed_force_last_remote_count" "$remote_count"

  if (( force_seed == 0 )) && (( planned_exec >= AUTO_SEED_PENDING_THRESHOLD )) && (( ready_depth >= READY_BUFFER_REFILL_THRESHOLD )); then
    return 0
  fi
  if (( now_epoch - last_seed < AUTO_SEED_COOLDOWN_SEC )); then
    return 0
  fi

  effective_seed_pending_threshold="$AUTO_SEED_PENDING_THRESHOLD"
  if (( force_seed == 1 )); then
    effective_seed_pending_threshold=$(( pending_exec + 1 ))
    if (( effective_seed_pending_threshold <= AUTO_SEED_PENDING_THRESHOLD )); then
      effective_seed_pending_threshold=$(( AUTO_SEED_PENDING_THRESHOLD + 1 ))
    fi
  fi

  local rc=0
  (
    cd "$ROOT_DIR"
    PYTHONPATH=src ./.venv/bin/python scripts/optimization/autonomous_queue_seeder.py \
      --pending-threshold "$effective_seed_pending_threshold" \
      --num-variants "$AUTO_SEED_NUM_VARIANTS" \
      --num-variants-floor "$AUTO_SEED_NUM_VARIANTS_FLOOR" \
      --aggregate-dir artifacts/wfa/aggregate \
      --run-index artifacts/wfa/aggregate/rollup/run_index.csv
  ) >>"$LOG_FILE" 2>&1 || rc=$?

  if (( rc == 0 )); then
    fullspan_state_metric_set "last_seed_trigger_epoch" "$now_epoch"
    fullspan_state_metric_set "seed_trigger_reason" "$reason"
    fullspan_state_metric_inc "seed_trigger_count" 1
    fullspan_state_metric_set "auto_seed_force_last_applied" "$force_seed"
    log "auto_seed_trigger reason=$reason force=$force_seed remote_count=$remote_count planned_exec=$planned_exec pending_exec=$pending_exec ready_depth=$ready_depth no_op_queues=$no_op_queues threshold=$effective_seed_pending_threshold num_variants=$AUTO_SEED_NUM_VARIANTS"
    log_decision_note "global" "AUTO_SEED_TRIGGER" "reason=$reason force=$force_seed remote_count=$remote_count planned_exec=$planned_exec pending_exec=$pending_exec ready_depth=$ready_depth threshold=$effective_seed_pending_threshold num_variants=$AUTO_SEED_NUM_VARIANTS" "expand_planned_backlog"
  else
    fullspan_state_metric_inc "seed_trigger_fail_count" 1
    log "auto_seed_failed reason=$reason force=$force_seed remote_count=$remote_count rc=$rc planned_exec=$planned_exec threshold=$effective_seed_pending_threshold"
  fi
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

vps_serverspace_state() {
  local api_script="$ROOT_DIR/scripts/vps/serverspace_api.py"
  local py="$ROOT_DIR/.venv/bin/python"
  local raw
  if [[ ! -f "$api_script" ]]; then
    echo ""
    return 0
  fi
  if [[ ! -x "$py" ]]; then
    py="python3"
  fi
  raw="$("$py" "$api_script" --ip "$SERVER_IP" find 2>>"$LOG_FILE" || true)"
  python3 - "$raw" <<'PY'
import json
import sys

raw = sys.argv[1] if len(sys.argv) > 1 else ""
try:
    obj = json.loads(raw)
except Exception:
    print("")
    raise SystemExit(0)
print(str(obj.get("state") or "").strip().lower())
PY
}

vps_recover_backoff_seconds() {
  local fail_streak="$1"
  local base="$VPS_RECOVER_BASE_COOLDOWN_SEC"
  local max_backoff="$VPS_RECOVER_MAX_COOLDOWN_SEC"
  local backoff
  local i

  [[ "$base" =~ ^[0-9]+$ ]] || base=300
  [[ "$max_backoff" =~ ^[0-9]+$ ]] || max_backoff=3600
  if (( base < 30 )); then
    base=30
  fi
  if (( max_backoff < base )); then
    max_backoff="$base"
  fi
  [[ "$fail_streak" =~ ^[0-9]+$ ]] || fail_streak=1
  if (( fail_streak < 1 )); then
    fail_streak=1
  fi

  backoff="$base"
  i=1
  while (( i < fail_streak )); do
    backoff=$((backoff * 2))
    if (( backoff >= max_backoff )); then
      backoff="$max_backoff"
      break
    fi
    i=$((i + 1))
  done

  echo "$backoff"
}

vps_next_retry_sleep_seconds() {
  local now_epoch next_epoch sleep_sec
  now_epoch="$(date +%s)"
  next_epoch="$(vps_runtime_next_retry_get)"
  [[ "$next_epoch" =~ ^[0-9]+$ ]] || next_epoch=0
  if (( next_epoch > now_epoch )); then
    sleep_sec=$((next_epoch - now_epoch))
  else
    sleep_sec=5
  fi
  [[ "$VPS_UNREACHABLE_SLEEP_CAP_SEC" =~ ^[0-9]+$ ]] || VPS_UNREACHABLE_SLEEP_CAP_SEC=300
  if (( sleep_sec > VPS_UNREACHABLE_SLEEP_CAP_SEC )); then
    sleep_sec="$VPS_UNREACHABLE_SLEEP_CAP_SEC"
  fi
  if (( sleep_sec < 5 )); then
    sleep_sec=5
  fi
  echo "$sleep_sec"
}

vps_force_power_cycle() {
  local reason="${1:-auto}"
  local api_script="$ROOT_DIR/scripts/vps/serverspace_api.py"
  local py="$ROOT_DIR/.venv/bin/python"
  local server_json server_id now_epoch
  local shutdown_wait boot_wait

  if [[ ! -f "$api_script" ]]; then
    log "vps_force_cycle_skip reason=$reason cause=api_script_missing"
    return 1
  fi
  if [[ ! -x "$py" ]]; then
    py="python3"
  fi

  server_json="$("$py" "$api_script" --ip "$SERVER_IP" find 2>>"$LOG_FILE" || true)"
  server_id="$(python3 - "$server_json" <<'PY'
import json
import sys

raw = sys.argv[1] if len(sys.argv) > 1 else ""
try:
    obj = json.loads(raw)
except Exception:
    print("")
    raise SystemExit(0)
print(str(obj.get("server_id") or obj.get("id") or "").strip())
PY
)"
  if [[ -z "$server_id" ]]; then
    log "vps_force_cycle_skip reason=$reason cause=server_id_unresolved"
    return 1
  fi

  now_epoch="$(date +%s)"
  vps_runtime_last_force_cycle_epoch="$now_epoch"
  vps_runtime_commit
  fullspan_state_metric_inc "vps_force_cycle_attempt_count" 1
  log "vps_force_cycle_attempt reason=$reason server_id=$server_id"

  "$py" "$api_script" shutdown --id "$server_id" >>"$LOG_FILE" 2>&1 || true
  shutdown_wait="$VPS_FORCE_CYCLE_SHUTDOWN_WAIT_SEC"
  [[ "$shutdown_wait" =~ ^[0-9]+$ ]] || shutdown_wait=20
  if (( shutdown_wait < 5 )); then
    shutdown_wait=5
  fi
  sleep "$shutdown_wait"

  if ! "$py" "$api_script" power-on --id "$server_id" >>"$LOG_FILE" 2>&1; then
    fullspan_state_metric_inc "vps_force_cycle_fail_count" 1
    log "vps_force_cycle_fail reason=$reason stage=power_on"
    return 1
  fi

  boot_wait="$VPS_FORCE_CYCLE_BOOT_WAIT_SEC"
  [[ "$boot_wait" =~ ^[0-9]+$ ]] || boot_wait=600
  if (( boot_wait < 60 )); then
    boot_wait=60
  fi
  local deadline=$(( $(date +%s) + boot_wait ))
  while (( $(date +%s) < deadline )); do
    if vps_is_reachable; then
      fullspan_state_metric_inc "vps_force_cycle_success_count" 1
      log "vps_force_cycle_success reason=$reason server_id=$server_id"
      return 0
    fi
    sleep 5
  done

  fullspan_state_metric_inc "vps_force_cycle_fail_count" 1
  log "vps_force_cycle_fail reason=$reason stage=ssh_wait timeout_sec=$boot_wait"
  return 1
}

ensure_vps_ready() {
  local reason="${1:-auto}"
  local now_epoch
  local unreachable_since fail_streak
  local backoff next_epoch down_for attempt_no timeout_sec base_cooldown
  local force_cycle_streak force_cycle_cooldown last_force_cycle
  local state_hint active_fastpath skip_remote_probe
  now_epoch="$(date +%s)"
  base_cooldown="$VPS_RECOVER_BASE_COOLDOWN_SEC"
  [[ "$base_cooldown" =~ ^[0-9]+$ ]] || base_cooldown=300
  if (( base_cooldown < 30 )); then
    base_cooldown=30
  fi

  [[ "$vps_runtime_fail_streak" =~ ^[0-9]+$ ]] || vps_runtime_fail_streak=0
  [[ "$vps_runtime_unreachable_since" =~ ^[0-9]+$ ]] || vps_runtime_unreachable_since=0
  [[ "$vps_runtime_last_recover_epoch" =~ ^[0-9]+$ ]] || vps_runtime_last_recover_epoch=0
  [[ "$vps_runtime_next_retry_epoch" =~ ^[0-9]+$ ]] || vps_runtime_next_retry_epoch=0
  [[ "$vps_runtime_last_force_cycle_epoch" =~ ^[0-9]+$ ]] || vps_runtime_last_force_cycle_epoch=0

  if vps_is_reachable; then
    unreachable_since="$vps_runtime_unreachable_since"
    if (( unreachable_since > 0 )); then
      down_for=$((now_epoch - unreachable_since))
      if (( down_for < 0 )); then
        down_for=0
      fi
      log "vps_recovered_after_outage reason=$reason down_for_sec=$down_for"
    fi
    vps_runtime_unreachable_since=0
    vps_runtime_fail_streak=0
    vps_runtime_next_retry_epoch=0
    vps_runtime_commit
    fullspan_state_metric_set "vps_infra_fail_closed" 0
    return 0
  fi

  if (( vps_runtime_next_retry_epoch > now_epoch )); then
    return 1
  fi

  if (( now_epoch - vps_runtime_last_recover_epoch < base_cooldown )); then
    vps_runtime_next_retry_epoch=$((vps_runtime_last_recover_epoch + base_cooldown))
    vps_runtime_commit
    return 1
  fi

  if (( vps_runtime_unreachable_since <= 0 )); then
    vps_runtime_unreachable_since="$now_epoch"
    vps_runtime_commit
  fi
  unreachable_since="$vps_runtime_unreachable_since"

  fail_streak="$vps_runtime_fail_streak"
  attempt_no=$((fail_streak + 1))
  force_cycle_streak="$VPS_FORCE_CYCLE_STREAK"
  [[ "$force_cycle_streak" =~ ^[0-9]+$ ]] || force_cycle_streak=3
  if (( force_cycle_streak < 2 )); then
    force_cycle_streak=2
  fi
  force_cycle_cooldown="$VPS_FORCE_CYCLE_COOLDOWN_SEC"
  [[ "$force_cycle_cooldown" =~ ^[0-9]+$ ]] || force_cycle_cooldown=1800
  if (( force_cycle_cooldown < 120 )); then
    force_cycle_cooldown=120
  fi
  last_force_cycle="$vps_runtime_last_force_cycle_epoch"
  timeout_sec="$VPS_RECOVER_TIMEOUT_SEC"
  [[ "$timeout_sec" =~ ^[0-9]+$ ]] || timeout_sec=420
  if (( timeout_sec < 90 )); then
    timeout_sec=90
  fi
  state_hint=""
  active_fastpath=0
  skip_remote_probe=0
  if is_truthy "$VPS_RECOVER_ACTIVE_SSH_DOWN_FASTPATH"; then
    state_hint="$(vps_serverspace_state)"
    if [[ "$state_hint" == "active" ]]; then
      active_fastpath=1
    fi
  fi

  if (( active_fastpath == 1 )); then
    if (( now_epoch - last_force_cycle >= force_cycle_cooldown )); then
      log "vps_recover_fastpath reason=$reason state=$state_hint action=force_cycle"
      if vps_force_power_cycle "${reason}:active_no_ssh"; then
        vps_runtime_fail_streak=0
        vps_runtime_next_retry_epoch=0
        vps_runtime_unreachable_since=0
        vps_runtime_commit
        fullspan_state_metric_set "vps_infra_fail_closed" 0
        log_decision_note "global" "VPS_FORCE_CYCLE_RECOVERED" "reason=$reason state=$state_hint mode=active_no_ssh" "resume_queue_dispatch"
        return 0
      fi
      skip_remote_probe=1
      log "vps_recover_fastpath_fail reason=$reason state=$state_hint action=skip_long_probe"
    else
      log "vps_recover_fastpath_deferred reason=$reason state=$state_hint cooldown_remaining=$((force_cycle_cooldown - (now_epoch - last_force_cycle)))"
    fi
  fi

  vps_runtime_last_recover_epoch="$now_epoch"
  vps_runtime_commit
  if (( skip_remote_probe == 0 )); then
    log "vps_recover_attempt reason=$reason attempt=$attempt_no timeout_sec=$timeout_sec state_hint=${state_hint:-unknown}"
    if timeout "$timeout_sec" env SKIP_POWER=0 STOP_AFTER=0 UPDATE_CODE=0 SYNC_BACK=0 SYNC_UP=0 "$ROOT_DIR/scripts/remote/run_server_job.sh" echo ping >>"$LOG_FILE" 2>&1 && vps_is_reachable; then
      fullspan_state_metric_inc "vps_recover_success_count" 1
      vps_runtime_fail_streak=0
      vps_runtime_next_retry_epoch=0
      vps_runtime_unreachable_since=0
      vps_runtime_commit
      fullspan_state_metric_set "vps_infra_fail_closed" 0
      log "vps_recover_success reason=$reason"
      return 0
    fi
  else
    log "vps_recover_skip_remote_probe reason=$reason state_hint=${state_hint:-unknown}"
  fi

  fail_streak=$((fail_streak + 1))
  backoff="$(vps_recover_backoff_seconds "$fail_streak")"
  [[ "$backoff" =~ ^[0-9]+$ ]] || backoff="$VPS_RECOVER_BASE_COOLDOWN_SEC"
  if (( backoff < 30 )); then
    backoff=30
  fi
  next_epoch=$((now_epoch + backoff))
  down_for=$((now_epoch - unreachable_since))
  if (( down_for < 0 )); then
    down_for=0
  fi

  vps_runtime_fail_streak="$fail_streak"
  vps_runtime_next_retry_epoch="$next_epoch"
  vps_runtime_commit
  fullspan_state_metric_set "vps_unreachable_duration_sec" "$down_for"
  fullspan_state_metric_inc "vps_recover_fail_count" 1

  if (( fail_streak >= force_cycle_streak )) && (( now_epoch - last_force_cycle >= force_cycle_cooldown )); then
    if vps_force_power_cycle "$reason"; then
      vps_runtime_fail_streak=0
      vps_runtime_next_retry_epoch=0
      vps_runtime_unreachable_since=0
      vps_runtime_commit
      fullspan_state_metric_set "vps_infra_fail_closed" 0
      log_decision_note "global" "VPS_FORCE_CYCLE_RECOVERED" "reason=$reason fail_streak=$fail_streak" "resume_queue_dispatch"
      return 0
    fi
    log_decision_note "global" "VPS_FORCE_CYCLE_FAIL" "reason=$reason fail_streak=$fail_streak next_retry_epoch=$next_epoch" "await_next_watchdog_cycle"
  fi

  if (( fail_streak >= VPS_RECOVER_FAIL_HARD_NOTE_STREAK )); then
    fullspan_state_metric_set "vps_infra_fail_closed" 1
    log_decision_note "global" "INFRA_FAIL_CLOSED" "reason=vps_unreachable fail_streak=$fail_streak down_for_sec=$down_for next_retry_epoch=$next_epoch" "continue_search_without_promote"
  fi
  log_decision_note "global" "VPS_RECOVER_FAIL" "reason=$reason attempt=$fail_streak down_for_sec=$down_for next_retry_epoch=$next_epoch" "await_next_watchdog_cycle"
  log "vps_recover_fail reason=$reason attempt=$fail_streak down_for_sec=$down_for backoff_sec=$backoff next_retry_epoch=$next_epoch"
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
  local candidate_uid="${7:-}"
  local top_run_group top_score

  if [[ -z "$top_variant" ]]; then
    return 0
  fi
  if (( strict_pass_count <= 0 || strict_run_groups < FULLSPAN_CONFIRM_MIN_GROUPS || confirm_count >= FULLSPAN_CONFIRM_MIN_REPLIES )); then
    return 0
  fi

  top_run_group="$(fullspan_state_get "$source_queue_rel" "top_run_group" "")"
  top_score="$(fullspan_state_get "$source_queue_rel" "top_score" "")"
  if [[ -z "$candidate_uid" ]]; then
    candidate_uid="$(derive_candidate_uid "$top_run_group" "$top_variant" "$top_score")"
  fi
  if [[ -z "$candidate_uid" ]]; then
    return 0
  fi

  local now_epoch last_trigger
  local dispatch_id
  now_epoch="$(date +%s)"
  dispatch_id="d${now_epoch}_$(printf '%s' "${candidate_uid}|${top_variant}|${top_run_group}" | sha1sum | awk '{print substr($1,1,10)}')"
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

def is_stress_row(config_path: str, results_dir: str) -> bool:
    cfg = str(config_path or '').strip().lower()
    if not cfg:
        return False
    cfg_name = Path(cfg).name
    cfg_stem = Path(cfg).stem
    if cfg.endswith('_stress.yaml'):
        return True
    if cfg_name.startswith('stress_') or cfg_stem.startswith('stress_'):
        return True

    results = str(results_dir or '').strip().lower().replace('\\', '/')
    if not results:
        return False
    segments = [segment for segment in results.split('/') if segment]
    run_id = segments[-1] if segments else ''
    if run_id.startswith('stress_'):
        return True
    if 'stress' in segments:
        return True
    return False

picked = []
seen = set()
for r in rows:
    cfg = str(r.get('config_path') or '').strip()
    results_dir = str(r.get('results_dir') or '').strip()
    if not cfg or cfg in seen:
        continue
    if is_stress_row(cfg, results_dir):
        continue
    blob = " ".join([cfg, results_dir, str(r.get('status') or '')]).lower()
    if needle and needle not in blob:
        continue
    seen.add(cfg)
    picked.append({'config_path': cfg, 'results_dir': results_dir, 'status': 'planned'})

if not picked:
    for r in rows:
        cfg = str(r.get('config_path') or '').strip()
        results_dir = str(r.get('results_dir') or '').strip()
        if not cfg or cfg in seen:
            continue
        if is_stress_row(cfg, results_dir):
            continue
        seen.add(cfg)
        picked.append({'config_path': cfg, 'results_dir': results_dir, 'status': 'planned'})
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
    "confirm_pending_since_epoch" "${now_epoch}" \
    "candidate_uid" "$candidate_uid" \
    "confirm_fastlane_last_dispatch_id" "$dispatch_id"

  (cd "$ROOT_DIR" && ./.venv/bin/python scripts/optimization/fullspan_lineage.py register \
      --registry "$CONFIRM_LINEAGE_REGISTRY_FILE" \
      --queue-rel "$confirm_queue_rel" \
      --source-queue-rel "$source_queue_rel" \
      --candidate-uid "$candidate_uid" \
      --top-run-group "$top_run_group" \
      --top-variant "$top_variant" \
      --dispatch-id "$dispatch_id") >>"$LOG_FILE" 2>&1 || true

  local qlog="$STATE_DIR/confirm_fastlane_${safe_name}_$(date -u +%Y%m%d_%H%M%S).log"
  local queue_poweroff
  queue_poweroff="$(batch_session_queue_poweroff)"
  if ! ensure_vps_ready "confirm_fastlane:$source_queue_rel"; then
    log "confirm_fastlane_deferred source=$source_queue_rel reason=vps_unreachable candidate_uid=$candidate_uid dispatch_id=$dispatch_id"
    log_decision_note "$source_queue_rel" "CONFIRM_FASTLANE_DEFERRED" "reason=vps_unreachable dispatch_id=$dispatch_id" "wait_vps_recovery"
    return 0
  fi
  (
    cd "$ROOT_DIR"
    # Do not leak driver lock fd into long-lived child workers.
    exec 9>&-
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
      --poweroff "$queue_poweroff" \
      >>"$qlog" 2>&1
  ) &
  batch_session_note_dispatch

  fullspan_state_metric_inc "confirm_fastlane_trigger_count" 1
  runtime_observability_record_event "winner_proximate_dispatch" "$now_epoch"
  log_decision_note "$source_queue_rel" "CONFIRM_FASTLANE_TRIGGER" "confirm_queue=$confirm_queue_rel variant=$top_variant candidate_uid=$candidate_uid dispatch_id=$dispatch_id" "await_confirm_replay"
  log "confirm_fastlane_trigger source=$source_queue_rel confirm_queue=$confirm_queue_rel variant=$top_variant candidate_uid=$candidate_uid dispatch_id=$dispatch_id"
}

dispatch_replay_fastlane_hooks() {
  if [[ "$DRIVER_CONFIRM_FASTLANE_ENABLE" != "1" && "$DRIVER_CONFIRM_FASTLANE_ENABLE" != "true" ]]; then
    return 0
  fi

  local dispatched=0
  while IFS=$'\t' read -r queue_rel queue_name top_variant strict_pass_count strict_run_groups confirm_count candidate_uid; do
    if [[ -z "$queue_rel" || -z "$top_variant" ]]; then
      continue
    fi
    trigger_confirm_fastlane "$queue_rel" "$queue_name" "$top_variant" "$strict_pass_count" "$strict_run_groups" "$confirm_count" "$candidate_uid"
    dispatched=$((dispatched + 1))
    if (( dispatched >= REPLAY_FASTLANE_SCAN_LIMIT )); then
      break
    fi
  done < <(
    python3 - "$FULLSPAN_DECISION_STATE_FILE" "$REPLAY_FASTLANE_SCAN_LIMIT" "$FULLSPAN_CONFIRM_MIN_GROUPS" "$FULLSPAN_CONFIRM_MIN_REPLIES" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
try:
    limit = max(1, int(float(sys.argv[2] or 2)))
except Exception:
    limit = 2
try:
    min_groups = max(1, int(float(sys.argv[3] or 2)))
except Exception:
    min_groups = 2
try:
    min_replies = max(1, int(float(sys.argv[4] or 2)))
except Exception:
    min_replies = 2
if not state_path.exists():
    raise SystemExit(0)
try:
    payload = json.loads(state_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)
queues = payload.get("queues", {}) if isinstance(payload, dict) else {}
if not isinstance(queues, dict):
    raise SystemExit(0)
ranked = []
for queue_rel, entry in queues.items():
    if not isinstance(entry, dict):
        continue
    verdict = str(entry.get("promotion_verdict") or "").strip().upper()
    if verdict not in {"PROMOTE_PENDING_CONFIRM", "PROMOTE_DEFER_CONFIRM"}:
        continue
    try:
        strict_pass_count = int(float(entry.get("strict_pass_count", 0) or 0))
    except Exception:
        strict_pass_count = 0
    try:
        strict_run_groups = int(float(entry.get("strict_run_group_count", 0) or 0))
    except Exception:
        strict_run_groups = 0
    try:
        confirm_count = int(float(entry.get("confirm_count", 0) or 0))
    except Exception:
        confirm_count = 0
    if strict_pass_count <= 0 or strict_run_groups < min_groups or confirm_count >= min_replies:
        continue
    top_variant = str(entry.get("top_variant") or "").strip()
    if not top_variant:
        continue
    queue_name = Path(str(queue_rel)).parent.name
    candidate_uid = str(entry.get("candidate_uid") or "").strip()
    try:
        pending_since = int(float(entry.get("confirm_pending_since_epoch", 0) or 0))
    except Exception:
        pending_since = 0
    ranked.append((pending_since if pending_since > 0 else 0, -strict_pass_count, str(queue_rel), queue_name, top_variant, strict_pass_count, strict_run_groups, confirm_count, candidate_uid))

for _, _, queue_rel, queue_name, top_variant, strict_pass_count, strict_run_groups, confirm_count, candidate_uid in sorted(ranked)[:limit]:
    print(f"{queue_rel}\t{queue_name}\t{top_variant}\t{strict_pass_count}\t{strict_run_groups}\t{confirm_count}\t{candidate_uid}")
PY
  )
  if (( dispatched > 0 )); then
    log "replay_fastlane_scan dispatched=$dispatched"
  fi
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
  local failed=0
  local completed=0
  local pending=0
  local executable_pending=0
  local completed_with_metrics=0

  local parallel
  local max_retries
  local queue_poweroff

  if is_truthy "$FORCE_SYNC_BEFORE_START"; then
    sync_queue_status "$queue_rel"
  fi

  read -r pending executable_pending completed_with_metrics planned running stalled failed completed <<< "$(queue_hygiene_snapshot "$queue_rel")"
  total=$((planned + running + stalled + failed + completed))

	  if (( pending <= 0 )); then
    log "queue_hygiene_empty_skip queue=$queue_rel pending=$pending completed_metrics=$completed_with_metrics"
    return 2
  fi

  if (( pending > 0 && executable_pending == 0 && completed_with_metrics > 0 )); then
    fullspan_state_metric_inc "no_op_queue_skips" 1
    fullspan_state_metric_set "last_no_op_queue_skip_epoch" "$(date +%s)"
    fullspan_state_metric_set "seed_trigger_reason" "queue_hygiene_noop"
    log "queue_hygiene_noop_skip queue=$queue_rel pending=$pending executable_pending=$executable_pending completed_metrics=$completed_with_metrics"
    log_decision_note "$queue_rel" "QUEUE_HYGIENE_NOOP_SKIP" "pending=$pending executable_pending=$executable_pending completed_metrics=$completed_with_metrics" "skip_and_reselect"
    return 2
  fi

  parallel="$(choose_parallel "$planned" "$running" "$stalled" "$total" "$cause")"
  max_retries="$(choose_max_retries "$cause")"
  queue_poweroff="$(batch_session_queue_poweroff)"

  local stamp
  stamp="$(date -u +%Y%m%d_%H%M%S)"
  local qlog="$STATE_DIR/run_${stamp}_$(basename "$(dirname "$queue_rel")").log"

  log "start queue_rel=$queue_rel cause=$cause parallel=$parallel max_retries=$max_retries"
  if ! ensure_vps_ready "start_queue:$queue_rel"; then
    log "start_blocked queue=$queue_rel reason=vps_unreachable"
    return 1
  fi
  (
    cd "$ROOT_DIR"
    # Do not leak driver lock fd into long-lived child workers.
    exec 9>&-
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
      --poweroff "$queue_poweroff" \
      >>"$qlog" 2>&1
  ) &
  batch_session_note_dispatch
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    log "failed_to_start queue=$queue_rel rc=$rc"
    return "$rc"
  fi
  log_state "running queue=$queue_rel reason=$cause started_at=$stamp log=$qlog parallel=$parallel max_retries=$max_retries selection_policy=$FULLSPAN_POLICY_NAME selection_mode=$PROMOTION_SELECTION_MODE promotion_verdict=$promotion_verdict pre_rank_score=$pre_rank_score promotion_potential=$promotion_potential gate_status=$gate_status effective_planned_count=${effective_planned_count:-0} stalled_share=${stalled_share:-0} queue_yield_score=${queue_yield_score:-0} recent_yield=${recent_yield:-0}"
  if [[ "$(queue_is_winner_proximate "$queue_rel")" == "1" ]]; then
    runtime_observability_record_event "winner_proximate_dispatch" "$stamp"
    fullspan_state_metric_inc "winner_proximate_dispatch_total" 1
  fi
  log "started queue=$queue_rel log=$qlog selection_policy=$FULLSPAN_POLICY_NAME selection_mode=$PROMOTION_SELECTION_MODE promotion_verdict=$promotion_verdict"
}

maybe_dispatch_overlap_from_buffer() {
  local active_queue_rel="$1"
  local active_pending="$2"
  local active_running="$3"
  local active_stalled="$4"
  local active_failed="$5"
  local active_total="$6"

  if (( active_pending <= 0 )); then
    return 1
  fi

  local active_exec_pending active_completed_metrics active_planned
  read -r _active_pending active_exec_pending active_completed_metrics active_planned _active_running _active_stalled _active_failed _active_completed <<< "$(queue_hygiene_snapshot "$active_queue_rel")"
  [[ "$active_exec_pending" =~ ^[0-9]+$ ]] || active_exec_pending=0
  if (( active_exec_pending > READY_BUFFER_OVERLAP_TAIL_PENDING )); then
    return 1
  fi

  local remote_jobs
  remote_jobs="$(remote_active_queue_jobs)"
  [[ "$remote_jobs" =~ ^[0-9]+$ ]] || remote_jobs=0
  refresh_remote_runtime_metrics
  if (( remote_jobs >= READY_BUFFER_MAX_ACTIVE_REMOTE_QUEUES )); then
    return 1
  fi

  ready_buffer_refresh "$active_queue_rel" || true
  local next_queue=""
  next_queue="$(ready_buffer_emit_candidate "$active_queue_rel" 2>/dev/null || true)"
  if [[ -z "$next_queue" ]]; then
    return 1
  fi

  IFS=',' read -r queue planned running stalled failed completed total urgency mtime promotion_potential gate_status gate_reason pre_rank_score strict_gate_status strict_gate_reason effective_planned_count stalled_share queue_yield_score recent_yield < <(tail -n 1 "$CANDIDATE_FILE")
  if [[ -z "${queue:-}" || "$queue" == "queue" ]]; then
    ready_buffer_release_claim "$next_queue"
    return 1
  fi

  local overlap_cause="UNKNOWN"
  if start_queue "$queue" "$overlap_cause" "${planned:-0}" "${running:-0}" "${stalled:-0}" "${total:-0}"; then
    fullspan_state_metric_inc "overlap_dispatch_count" 1
    log_decision_note "$active_queue_rel" "READY_BUFFER_DISPATCH" "next_queue=$queue tail_exec_pending=$active_exec_pending remote_active_queue_jobs=$remote_jobs" "dispatch_next_queue_overlap"
    log "ready_buffer_dispatch source=$active_queue_rel next_queue=$queue tail_exec_pending=$active_exec_pending remote_active_queue_jobs=$remote_jobs"
    return 0
  fi

  ready_buffer_release_claim "$queue"
  return 1
}

# lightweight per-queue progress cache for stale detection in-shell
declare -A last_pending_by_queue

cleanup_orphans
batch_session_load_state
vps_runtime_load
vps_runtime_commit

declare -A stalled_repair_failures
declare -A stalled_repair_last_reason_by_queue
declare -A stalled_repair_same_reason_streak_by_queue
declare -A no_progress_streak_by_queue
declare -A vps_idle_pending_streak_by_queue
declare -A vps_unreachable_streak_by_queue

adaptive_idle_sleep=30
prev_queue=""
prev_planned=0
prev_running=0
prev_stalled=0
prev_failed=0
prev_pending=0
busy_repeat_count=0
busy_backoff_seconds=90
no_progress_breaker_streak_count="$(fullspan_state_metric_get no_progress_breaker_streak 0)"
[[ "$no_progress_breaker_streak_count" =~ ^[0-9]+$ ]] || no_progress_breaker_streak_count=0
no_progress_breaker_last_progress_epoch="$(fullspan_state_metric_get last_completed_with_metrics_progress_epoch 0)"
[[ "$no_progress_breaker_last_progress_epoch" =~ ^[0-9]+$ ]] || no_progress_breaker_last_progress_epoch=0
global_completed_metrics_last="$(fullspan_state_metric_get last_completed_with_metrics_count 0)"
[[ "$global_completed_metrics_last" =~ ^[0-9]+$ ]] || global_completed_metrics_last=0
initial_completed_metrics="$(global_completed_metrics_count)"
[[ "$initial_completed_metrics" =~ ^[0-9]+$ ]] || initial_completed_metrics=0
if (( initial_completed_metrics > global_completed_metrics_last )); then
  global_completed_metrics_last="$initial_completed_metrics"
fi
if (( no_progress_breaker_last_progress_epoch <= 0 )); then
  no_progress_breaker_last_progress_epoch="$(date +%s)"
fi
fullspan_state_metric_set "last_completed_with_metrics_count" "$global_completed_metrics_last"
fullspan_state_metric_set "last_completed_with_metrics_progress_epoch" "$no_progress_breaker_last_progress_epoch"
fullspan_state_metric_set "no_progress_breaker_streak" "$no_progress_breaker_streak_count"
fullspan_state_metric_set "completed_with_metrics_global" "$global_completed_metrics_last"

if ! find "$QUEUE_ROOT" -name run_queue.csv >/dev/null 2>&1; then
  log_state "No aggregate queues found: $QUEUE_ROOT"
  exit 1
fi

while true; do
  maybe_trigger_auto_seed "low_planned_backlog" || true
  fullspan_state_metric_set "ready_buffer_depth" "$(ready_buffer_depth)"
  fullspan_state_metric_set "cold_fail_active_count" "$(cold_fail_active_count)"
  refresh_remote_runtime_metrics
  refresh_runtime_observability_metrics
  dispatch_replay_fastlane_hooks || true
  maybe_prepare_hot_standby || true

  current_epoch="$(date +%s)"
  global_completed_metrics_now="$(global_completed_metrics_count)"
  [[ "$global_completed_metrics_now" =~ ^[0-9]+$ ]] || global_completed_metrics_now=0
  fullspan_state_metric_set "completed_with_metrics_global" "$global_completed_metrics_now"
  phase_switch_required=0

  if (( global_completed_metrics_now > global_completed_metrics_last )); then
    delta_completed_metrics=$((global_completed_metrics_now - global_completed_metrics_last))
    global_completed_metrics_last="$global_completed_metrics_now"
    no_progress_breaker_streak_count=0
    no_progress_breaker_last_progress_epoch="$current_epoch"
    fullspan_state_metric_set "last_completed_with_metrics_count" "$global_completed_metrics_last"
    fullspan_state_metric_set "last_completed_with_metrics_progress_epoch" "$no_progress_breaker_last_progress_epoch"
    fullspan_state_metric_set "no_progress_breaker_streak" "$no_progress_breaker_streak_count"
    fullspan_state_metric_set "completed_with_metrics_stagnant_sec" "0"
    log "global_progress completed_with_metrics=$global_completed_metrics_now delta=$delta_completed_metrics"
  else
    stagnant_sec=$((current_epoch - no_progress_breaker_last_progress_epoch))
    if (( stagnant_sec < 0 )); then
      stagnant_sec=0
    fi
    fullspan_state_metric_set "completed_with_metrics_stagnant_sec" "$stagnant_sec"
    if (( stagnant_sec >= NO_PROGRESS_BREAKER_WINDOW_SEC )); then
      no_progress_breaker_streak_count=$((no_progress_breaker_streak_count + 1))
      no_progress_breaker_last_progress_epoch="$current_epoch"
      fullspan_state_metric_set "last_completed_with_metrics_progress_epoch" "$no_progress_breaker_last_progress_epoch"
      fullspan_state_metric_set "no_progress_breaker_streak" "$no_progress_breaker_streak_count"
      log "no_progress_breaker_window stagnant_sec=$stagnant_sec streak=$no_progress_breaker_streak_count window_sec=$NO_PROGRESS_BREAKER_WINDOW_SEC completed_with_metrics=$global_completed_metrics_now"
      if (( no_progress_breaker_streak_count >= NO_PROGRESS_BREAKER_STREAK )); then
        phase_switch_required=1
      fi
    fi
  fi

  find_candidate 1
  ready_buffer_refresh "" || true

  if [[ ! -s "$CANDIDATE_FILE" ]]; then
    cleanup_orphans
    find_candidate 0
    ready_buffer_refresh "" || true
  fi

  if [[ ! -s "$CANDIDATE_FILE" ]]; then
    if ready_buffer_emit_candidate "${LAST_REJECTED_QUEUE:-}" >/dev/null 2>&1; then
      log "ready_buffer_hit reason=no_candidate_file"
      log_decision_note "global" "READY_BUFFER_HIT" "reason=no_candidate_file" "reuse_ready_buffer_candidate"
    fi
  fi

  if [[ ! -s "$CANDIDATE_FILE" ]]; then
    if fallback_pending_candidate "$ORPHAN_FILE" "$LAST_REJECTED_QUEUE" "$FULLSPAN_DECISION_STATE_FILE"; then
      log "candidate_fallback_selected reason=no_candidate_file"
	    else
	      log_state "idle now=none completed=all"
	      log "candidate_empty"
	      maybe_trigger_auto_seed "candidate_empty" || true
	      if [[ "$adaptive_idle_sleep" -lt 300 ]]; then
	        adaptive_idle_sleep=$((adaptive_idle_sleep * 2))
	      fi
      if [[ "$adaptive_idle_sleep" -gt 300 ]]; then
        adaptive_idle_sleep=300
      fi
      batch_session_maybe_stop "candidate_empty"
      sleep "$adaptive_idle_sleep"
      continue
    fi
  fi

  IFS=',' read -r queue planned running stalled failed completed total urgency mtime promotion_potential gate_status gate_reason pre_rank_score strict_gate_status strict_gate_reason effective_planned_count stalled_share queue_yield_score recent_yield < <(tail -n 1 "$CANDIDATE_FILE")

  if [[ -z "${queue:-}" || "$queue" == "queue" ]]; then
    if fallback_pending_candidate "$ORPHAN_FILE" "$LAST_REJECTED_QUEUE" "$FULLSPAN_DECISION_STATE_FILE"; then
      log "candidate_parse_fallback reason=parse_empty"
      IFS=',' read -r queue planned running stalled failed completed total urgency mtime promotion_potential gate_status gate_reason pre_rank_score strict_gate_status strict_gate_reason effective_planned_count stalled_share queue_yield_score recent_yield < <(tail -n 1 "$CANDIDATE_FILE")
    fi
  fi

	  if [[ -z "${queue:-}" || "$queue" == "queue" ]]; then
      ready_buffer_refresh "${LAST_REJECTED_QUEUE:-}" || true
      if ready_buffer_emit_candidate "${LAST_REJECTED_QUEUE:-}" >/dev/null 2>&1; then
        log "ready_buffer_hit reason=candidate_parse_empty"
        log_decision_note "global" "READY_BUFFER_HIT" "reason=candidate_parse_empty" "reuse_ready_buffer_candidate"
        IFS=',' read -r queue planned running stalled failed completed total urgency mtime promotion_potential gate_status gate_reason pre_rank_score strict_gate_status strict_gate_reason effective_planned_count stalled_share queue_yield_score recent_yield < <(tail -n 1 "$CANDIDATE_FILE")
      else
	    log_state "idle now=none completed=all"
	    log "candidate_parse_empty"
	    maybe_trigger_auto_seed "candidate_parse_empty" || true
	    if [[ "$adaptive_idle_sleep" -lt 300 ]]; then
	      adaptive_idle_sleep=$((adaptive_idle_sleep * 2))
	    fi
        if [[ "$adaptive_idle_sleep" -gt 300 ]]; then
          adaptive_idle_sleep=300
        fi
        batch_session_maybe_stop "candidate_parse_empty"
        sleep "$adaptive_idle_sleep"
        continue
      fi
	  fi

	  queue_rel="${queue#$ROOT_DIR/}"
	  recent_yield="${recent_yield:-0}"
	  fullspan_state_metric_set "recent_yield" "$recent_yield"
	  pending=$((planned + running + stalled + failed))

  queue_abs="$ROOT_DIR/$queue_rel"
  if [[ -f "$queue_abs" ]]; then
    read -r q_planned q_running q_stalled q_failed q_completed q_total <<< "$(python3 - "$queue_abs" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    print("0 0 0 0 0 0")
    raise SystemExit(0)

rows = list(csv.DictReader(p.open(newline='', encoding='utf-8')))
c = Counter((r.get('status') or '').strip().lower() for r in rows)
planned = int(c.get('planned', 0))
running = int(c.get('running', 0))
stalled = int(c.get('stalled', 0))
failed = int(c.get('failed', 0)) + int(c.get('error', 0))
completed = int(c.get('completed', 0))
print(planned, running, stalled, failed, completed, len(rows))
PY
)"

    if [[ "$q_planned" != "$planned" || "$q_running" != "$running" || "$q_stalled" != "$stalled" || "$q_failed" != "$failed" || "$q_completed" != "$completed" || "$q_total" != "$total" ]]; then
      log "candidate_reconcile queue=$queue_rel cand=${planned}/${running}/${stalled}/${failed}/${completed}/${total} actual=${q_planned}/${q_running}/${q_stalled}/${q_failed}/${q_completed}/${q_total}"
      planned="$q_planned"
      running="$q_running"
      stalled="$q_stalled"
      failed="$q_failed"
      completed="$q_completed"
      total="$q_total"
      pending=$((planned + running + stalled + failed))
    fi
  fi

  if (( pending <= 0 )); then
    : > "$CANDIDATE_FILE"
    find_candidate 1
    if [[ ! -s "$CANDIDATE_FILE" ]]; then
      cleanup_orphans
      find_candidate 0
    fi
    if [[ ! -s "$CANDIDATE_FILE" ]]; then
      fallback_pending_candidate "$ORPHAN_FILE" "$LAST_REJECTED_QUEUE" "$FULLSPAN_DECISION_STATE_FILE" || true
    fi
	    if [[ ! -s "$CANDIDATE_FILE" ]]; then
	      log_state "idle now=none completed=all"
	      log "candidate_empty_after_reconcile"
	      maybe_trigger_auto_seed "candidate_empty_after_reconcile" || true
	      if [[ "$adaptive_idle_sleep" -lt 300 ]]; then
	        adaptive_idle_sleep=$((adaptive_idle_sleep * 2))
	      fi
      if [[ "$adaptive_idle_sleep" -gt 300 ]]; then
        adaptive_idle_sleep=300
      fi
      batch_session_maybe_stop "candidate_empty_after_reconcile"
      sleep "$adaptive_idle_sleep"
      continue
    fi
	    IFS=',' read -r queue planned running stalled failed completed total urgency mtime promotion_potential gate_status gate_reason pre_rank_score strict_gate_status strict_gate_reason effective_planned_count stalled_share queue_yield_score recent_yield < <(tail -n 1 "$CANDIDATE_FILE")
	    if [[ -z "${queue:-}" || "$queue" == "queue" ]]; then
	      if fallback_pending_candidate "$ORPHAN_FILE" "$LAST_REJECTED_QUEUE" "$FULLSPAN_DECISION_STATE_FILE"; then
	        log "candidate_parse_fallback reason=parse_empty_after_reconcile"
	        IFS=',' read -r queue planned running stalled failed completed total urgency mtime promotion_potential gate_status gate_reason pre_rank_score strict_gate_status strict_gate_reason effective_planned_count stalled_share queue_yield_score recent_yield < <(tail -n 1 "$CANDIDATE_FILE")
	      fi
	    fi
	    if [[ -z "${queue:-}" || "$queue" == "queue" ]]; then
        ready_buffer_refresh "${LAST_REJECTED_QUEUE:-}" || true
        if ready_buffer_emit_candidate "${LAST_REJECTED_QUEUE:-}" >/dev/null 2>&1; then
          log "ready_buffer_hit reason=candidate_parse_empty_after_reconcile"
          log_decision_note "global" "READY_BUFFER_HIT" "reason=candidate_parse_empty_after_reconcile" "reuse_ready_buffer_candidate"
          IFS=',' read -r queue planned running stalled failed completed total urgency mtime promotion_potential gate_status gate_reason pre_rank_score strict_gate_status strict_gate_reason effective_planned_count stalled_share queue_yield_score recent_yield < <(tail -n 1 "$CANDIDATE_FILE")
        else
	      log_state "idle now=none completed=all"
	      log "candidate_parse_empty_after_reconcile"
	      maybe_trigger_auto_seed "candidate_parse_empty_after_reconcile" || true
	      if [[ "$adaptive_idle_sleep" -lt 300 ]]; then
	        adaptive_idle_sleep=$((adaptive_idle_sleep * 2))
	      fi
	      if [[ "$adaptive_idle_sleep" -gt 300 ]]; then
	        adaptive_idle_sleep=300
	      fi
	      batch_session_maybe_stop "candidate_parse_empty_after_reconcile"
	      sleep "$adaptive_idle_sleep"
	      continue
        fi
	    fi
	    queue_rel="${queue#$ROOT_DIR/}"
	    recent_yield="${recent_yield:-0}"
	    fullspan_state_metric_set "recent_yield" "$recent_yield"
	    pending=$((planned + running + stalled + failed))
	    log "candidate_reselect_after_reconcile queue=$queue_rel pending=$pending"
	  fi

	  if (( phase_switch_required == 1 && pending > 0 && running == 0 )); then
	    queue_age_sec=999999
	    if [[ "${mtime:-}" =~ ^[0-9]+$ ]]; then
	      queue_age_sec=$((current_epoch - mtime))
	      if (( queue_age_sec < 0 )); then
	        queue_age_sec=0
	      fi
	    fi
	    if (( queue_age_sec < NO_PROGRESS_BREAKER_FRESH_QUEUE_GRACE_SEC && completed == 0 && failed == 0 && stalled == 0 && pending == planned )); then
	      no_progress_breaker_streak_count=0
	      fullspan_state_metric_set "no_progress_breaker_streak" "$no_progress_breaker_streak_count"
	      log "no_progress_phase_switch_deferred queue=$queue_rel pending=$pending reason=fresh_queue_grace age_sec=$queue_age_sec grace_sec=$NO_PROGRESS_BREAKER_FRESH_QUEUE_GRACE_SEC"
	    elif remote_queue_running "$queue_rel" || local_powered_queue_running "$queue_rel"; then
	      no_progress_breaker_streak_count=0
	      fullspan_state_metric_set "no_progress_breaker_streak" "$no_progress_breaker_streak_count"
	      log "no_progress_phase_switch_deferred queue=$queue_rel pending=$pending reason=queue_runner_active"
	    else
	    no_progress_breaker_streak_count=0
	    fullspan_state_metric_set "no_progress_breaker_streak" "$no_progress_breaker_streak_count"
	    fullspan_state_metric_inc "no_progress_breaker_trigger_count" 1
	    fullspan_state_metric_set "seed_trigger_reason" "no_progress_breaker"
	    LAST_REJECTED_QUEUE="$queue_rel"
	    mark_orphan "$queue_rel" "no_progress_breaker_phase_switch"
	    log_decision_note "$queue_rel" "NO_PROGRESS_PHASE_SWITCH" "pending=$pending planned=$planned stalled=$stalled failed=$failed recent_yield=${recent_yield:-0}" "orphan_and_reselect"
	    log "no_progress_phase_switch queue=$queue_rel pending=$pending planned=$planned stalled=$stalled failed=$failed recent_yield=${recent_yield:-0}"
	    maybe_trigger_auto_seed "no_progress_breaker" || true
	    : > "$CANDIDATE_FILE"
	    sleep 2
	    continue
	    fi
	  fi

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

  if (( planned > 0 && running == 0 )); then
    early_stop_payload="$(early_stop_low_yield_queue "$queue_rel")"
    early_stop_trigger="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(int(bool(d.get(\"trigger\", False))))' "$early_stop_payload" 2>/dev/null || echo 0)"
    if [[ "$early_stop_trigger" == "1" ]]; then
      early_stop_reason="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(str(d.get(\"reason\") or \"EARLY_ABORT_LOW_INFORMATION\"))' "$early_stop_payload" 2>/dev/null || echo EARLY_ABORT_LOW_INFORMATION)"
      early_stop_changed="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(int(d.get(\"changed\",0)))' "$early_stop_payload" 2>/dev/null || echo 0)"
      early_stop_fail_fraction="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(float(d.get(\"fail_fraction\",0.0)))' "$early_stop_payload" 2>/dev/null || echo 0)"
      early_stop_zero_activity_fraction="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(float(d.get(\"zero_activity_fraction\",0.0)))' "$early_stop_payload" 2>/dev/null || echo 0)"
      state_strict_pass_count="$(fullspan_state_get "$queue_rel" "strict_pass_count" "0")"
      state_strict_run_groups="$(fullspan_state_get "$queue_rel" "strict_run_group_count" "0")"
      state_confirm_count="$(fullspan_state_get "$queue_rel" "confirm_count" "0")"
      state_run_groups_csv="$(fullspan_state_run_groups_csv "$queue_rel")"
      state_strict_summary_path="$(fullspan_state_get "$queue_rel" "strict_summary_path" "")"
      fullspan_state_set "$queue_rel" "REJECT" \
        "$state_strict_pass_count" "$state_strict_run_groups" "" "" "" \
        "$early_stop_reason" "$state_confirm_count" "FULLSPAN_PREFILTER_REJECT" "$early_stop_reason" "$state_run_groups_csv" "$state_strict_summary_path"
      mark_orphan "$queue_rel" "early_stop_low_yield"
      if [[ "$early_stop_reason" == "EARLY_ABORT_ZERO_ACTIVITY" ]]; then
        fullspan_state_metric_inc "early_abort_zero_activity_count" 1
      else
        fullspan_state_metric_inc "early_abort_low_information_count" 1
      fi
      if [[ "$early_stop_reason" == "EARLY_ABORT_ZERO_ACTIVITY" || "$early_stop_reason" == *"METRICS_MISSING"* ]]; then
        runtime_observability_record_event "metrics_missing_abort"
      fi
      log_decision_note "$queue_rel" "${early_stop_reason}" "reason=$early_stop_reason changed=$early_stop_changed fail_fraction=$early_stop_fail_fraction zero_activity_fraction=$early_stop_zero_activity_fraction" "skip_remaining_and_continue_search"
      log "early_abort queue=$queue_rel reason=$early_stop_reason changed=$early_stop_changed fail_fraction=$early_stop_fail_fraction zero_activity_fraction=$early_stop_zero_activity_fraction"
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
      if (( pending <= 0 )); then
        sleep 2
        continue
      fi
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
  state_candidate_uid="$(fullspan_state_get "$queue_rel" "candidate_uid" "")"
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
  state_candidate_uid="$(fullspan_state_get "$queue_rel" "candidate_uid" "")"

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

  if [[ "$DRIVER_CONFIRM_FASTLANE_ENABLE" == "1" || "$DRIVER_CONFIRM_FASTLANE_ENABLE" == "true" ]]; then
    trigger_confirm_fastlane "$queue_rel" "$(basename "$(dirname "$queue_rel")")" "$state_top_variant" "$state_strict_pass_count" "$state_strict_run_groups" "$state_confirm_count" "$state_candidate_uid"
  fi

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
      if ensure_vps_ready "queue_start:$queue_rel"; then
        vps_unreachable_streak_by_queue["$queue_rel"]=0
        log "vps_recovered queue=$queue_rel pending=$pending"
      else
        unreachable_streak=0
        sleep_sec=5
        next_retry_epoch=0
        no_progress_streak_by_queue["$queue_rel"]=0
        unreachable_streak="${vps_unreachable_streak_by_queue[$queue_rel]:-0}"
        unreachable_streak=$((unreachable_streak + 1))
        vps_unreachable_streak_by_queue["$queue_rel"]="$unreachable_streak"
        fullspan_state_metric_inc "vps_unreachable_loop_count" 1
        next_retry_epoch="$(vps_runtime_next_retry_get)"
        [[ "$next_retry_epoch" =~ ^[0-9]+$ ]] || next_retry_epoch=0
        sleep_sec="$(vps_next_retry_sleep_seconds)"
        if (( unreachable_streak >= VPS_UNREACHABLE_ORPHAN_STREAK )); then
          LAST_REJECTED_QUEUE="$queue_rel"
          mark_orphan "$queue_rel" "vps_unreachable_streak_${unreachable_streak}"
          : > "$CANDIDATE_FILE"
          vps_unreachable_streak_by_queue["$queue_rel"]=0
          log_decision_note "$queue_rel" "INFRA_FAIL_CLOSED" "reason=vps_unreachable streak=$unreachable_streak next_retry_epoch=$next_retry_epoch" "orphan_and_continue_search"
        fi
        log "no_progress_pause queue=$queue_rel reason=vps_unreachable pending=$pending streak=$unreachable_streak next_retry_epoch=$next_retry_epoch sleep_sec=$sleep_sec"
        sleep "$sleep_sec"
        continue
      fi
    else
      vps_unreachable_streak_by_queue["$queue_rel"]=0
    fi

    if remote_queue_running "$queue_rel" || local_powered_queue_running "$queue_rel"; then
      no_progress_streak_by_queue["$queue_rel"]=0
      if remote_queue_running "$queue_rel"; then
        log "no_progress_pause queue=$queue_rel reason=remote_runner_active_sync pending=$pending"
      else
        log "no_progress_pause queue=$queue_rel reason=local_powered_runner_active pending=$pending"
      fi
      sync_queue_status "$queue_rel"
      if (( completed > 0 )); then
        fullspan_rollup_sync "$queue_rel" "remote_runner_active_sync"
      else
        log "fullspan_rollup_sync_skip queue=$queue_rel reason=remote_runner_active_sync completed=0"
      fi
      maybe_dispatch_overlap_from_buffer "$queue_rel" "$pending" "$running" "$stalled" "$failed" "$total" || true
      sleep 3
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
    LAST_REJECTED_QUEUE="$queue_rel"
    mark_orphan "$queue_rel" "gated_reject_no_progress"
    cold_fail_state_add "$queue_rel" "${gate_reason:-HARD_FAIL}"
    fullspan_state_metric_set "cold_fail_active_count" "$(cold_fail_active_count)"
    : > "$CANDIDATE_FILE"
    log "candidate_gated_reject queue=$queue_rel promotion_verdict=$promotion_verdict gate_status=$gate_status gate_reason=$gate_reason pre_rank_score=$pre_rank_score"
    log_decision_note "$queue_rel" "REJECT" "gate_status=$gate_status reason=$gate_reason" "skip_and_select_next_candidate"
    log_decision_note "$queue_rel" "COLD_FAIL_INDEX_ADD" "gate_reason=${gate_reason:-HARD_FAIL} ttl_sec=$HARD_FAIL_COLD_TTL_SEC" "exclude_from_hot_selector"
    batch_session_maybe_stop "gated_reject"
    sleep 2
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
    log "busy skip start queue_rel=$queue_rel urgency=$urgency reason=$reason pending=$pending repeat=$busy_repeat_count cause=$cause action=$action busy_reason=${busy_reason:-unknown}"
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
    batch_session_maybe_stop "no_pending"
    sleep "$adaptive_idle_sleep"
    continue
  fi

  refresh_gate_surrogate_state "$queue_rel" || true

  surrogate_decision="allow"
  surrogate_reason=""
  IFS=$'\t' read -r surrogate_decision surrogate_reason < <(surrogate_gate_decision "$queue_rel")
  surrogate_decision="$(printf '%s' "${surrogate_decision:-allow}" | tr '[:upper:]' '[:lower:]')"
  if [[ -z "$surrogate_decision" ]]; then
    surrogate_decision="allow"
  fi

  remote_active_jobs="$(remote_active_queue_jobs)"
  [[ "$remote_active_jobs" =~ ^[0-9]+$ ]] || remote_active_jobs=0
  idle_with_pending="$(process_slo_idle_with_executable_pending)"
  [[ "$idle_with_pending" =~ ^[0-9]+$ ]] || idle_with_pending=0
  refresh_remote_runtime_metrics
  if [[ "$surrogate_decision" == "refine" && "${surrogate_reason:-}" == "queue_pending_backlog" && "$planned" -gt 0 && "$running" -eq 0 && "$stalled" -eq 0 && "$failed" -eq 0 && "$completed" -eq 0 ]]; then
    if (( remote_active_jobs == 0 || idle_with_pending == 1 )); then
      surrogate_decision="allow"
      surrogate_reason="cold_start_idle_slot"
      fullspan_state_metric_inc "surrogate_idle_override_count" 1
      log "surrogate_gate queue=$queue_rel action=SURROGATE_IDLE_OVERRIDE reason=cold_start_idle_slot remote_active_queue_jobs=$remote_active_jobs idle_with_pending=$idle_with_pending"
      log_decision_note "$queue_rel" "SURROGATE_IDLE_OVERRIDE" "reason=cold_start_idle_slot remote_active_queue_jobs=$remote_active_jobs idle_with_pending=$idle_with_pending" "continue_dispatch"
    fi
  fi

  if [[ "$surrogate_decision" == "reject" ]]; then
    LAST_REJECTED_QUEUE="$queue_rel"
    mark_orphan "$queue_rel" "surrogate_reject"
    fullspan_state_queue_set "$queue_rel" \
      "promotion_verdict" "REJECT" \
      "rejection_reason" "surrogate_reject" \
      "strict_gate_status" "FULLSPAN_PREFILTER_REJECT" \
      "strict_gate_reason" "${surrogate_reason:-surrogate_reject}"
    fullspan_state_metric_inc "surrogate_reject_count" 1
    log "surrogate_gate queue=$queue_rel action=SURROGATE_REJECT reason=${surrogate_reason:-surrogate_reject}"
    log_decision_note "$queue_rel" "SURROGATE_REJECT" "reason=${surrogate_reason:-surrogate_reject}" "skip_and_fail_closed"
    : > "$CANDIDATE_FILE"
    batch_session_maybe_stop "surrogate_reject"
    sleep 2
    continue
  fi

  if [[ "$surrogate_decision" == "refine" ]]; then
    mark_orphan "$queue_rel" "surrogate_refine"
    fullspan_state_metric_inc "surrogate_refine_count" 1
    fullspan_state_metric_set "seed_trigger_reason" "surrogate_refine"
    log "surrogate_gate queue=$queue_rel action=SURROGATE_REFINE reason=${surrogate_reason:-surrogate_refine}"
    log_decision_note "$queue_rel" "SURROGATE_REFINE" "reason=${surrogate_reason:-surrogate_refine}" "skip_heavy_dispatch_and_seed"
    maybe_trigger_auto_seed "surrogate_refine" || true
    : > "$CANDIDATE_FILE"
    sleep 2
    continue
  fi

  if [[ "$surrogate_decision" == "allow" && -n "$surrogate_reason" ]]; then
    fullspan_state_metric_inc "surrogate_allow_count" 1
    log "surrogate_gate queue=$queue_rel action=SURROGATE_ALLOW reason=$surrogate_reason"
    log_decision_note "$queue_rel" "SURROGATE_ALLOW" "reason=$surrogate_reason" "continue_dispatch"
  fi

  if [[ -z "$cause" ]]; then
    cause="UNKNOWN"
  fi

  log "candidate queue=$queue_rel reason=$reason cause=$cause urgency=$urgency planned=$planned running=$running stalled=$stalled failed=$failed completed=$completed action=$action selection_policy=$FULLSPAN_POLICY_NAME selection_mode=$PROMOTION_SELECTION_MODE selection_profile=$PROMOTION_SELECTION_PROFILE promotion_verdict=$promotion_verdict gate_status=$gate_status gate_reason=$gate_reason promotion_potential=$promotion_potential pre_rank_score=$pre_rank_score effective_planned_count=${effective_planned_count:-0} stalled_share=${stalled_share:-0} queue_yield_score=${queue_yield_score:-0} recent_yield=${recent_yield:-0}"
  if start_queue "$queue_rel" "$cause" "$planned" "$running" "$stalled" "$total"; then
    start_rc=0
  else
    start_rc=$?
  fi
  if (( start_rc == 2 )); then
    fullspan_state_metric_set "seed_trigger_reason" "queue_hygiene_skip"
    log "start_skipped queue=$queue_rel cause=$cause reason=queue_hygiene_skip"
    maybe_trigger_auto_seed "queue_hygiene_skip" || true
    : > "$CANDIDATE_FILE"
    sleep 2
    continue
  fi
  if (( start_rc != 0 )); then
    log "start_failed queue=$queue_rel cause=$cause rc=$start_rc"
    batch_session_maybe_stop "start_failed"
  fi

  sleep 120

done
