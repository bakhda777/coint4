#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
STATE_DIR="$ROOT_DIR/artifacts/wfa/aggregate/.autonomous"
LOG_FILE="$STATE_DIR/confirm_dispatch_agent.log"
LOCK_FILE="$STATE_DIR/confirm_dispatch_agent.lock"
FULLSPAN_DECISION_STATE_FILE="$STATE_DIR/fullspan_decision_state.json"
DECISION_NOTES_FILE="$STATE_DIR/decision_notes.jsonl"
CONFIRM_LINEAGE_REGISTRY_FILE="$STATE_DIR/confirm_lineage_registry.json"
CAPACITY_CONTROLLER_STATE_FILE="${CAPACITY_CONTROLLER_STATE_FILE:-$STATE_DIR/capacity_controller_state.json}"
SERVER_IP="${SERVER_IP:-85.198.90.128}"
SERVER_USER="${SERVER_USER:-root}"
FULLSPAN_CONFIRM_MIN_GROUPS="${FULLSPAN_CONFIRM_MIN_GROUPS:-2}"
FULLSPAN_CONFIRM_MIN_REPLIES="${FULLSPAN_CONFIRM_MIN_REPLIES:-2}"
CONFIRM_FASTLANE_LIMIT="${CONFIRM_FASTLANE_LIMIT:-1}"
CONFIRM_FASTLANE_PARALLEL="${CONFIRM_FASTLANE_PARALLEL:-2}"
CONFIRM_FASTLANE_MAX_LIMIT="${CONFIRM_FASTLANE_MAX_LIMIT:-3}"
CONFIRM_FASTLANE_MAX_PARALLEL="${CONFIRM_FASTLANE_MAX_PARALLEL:-4}"
CONFIRM_FASTLANE_COOLDOWN_SEC="${CONFIRM_FASTLANE_COOLDOWN_SEC:-1800}"
CONFIRM_DISPATCHES_PER_CYCLE="${CONFIRM_DISPATCHES_PER_CYCLE:-2}"
CONFIRM_LANE_MAX_ACTIVE="${CONFIRM_LANE_MAX_ACTIVE:-2}"
CONFIRM_LANE_MAX_REMOTE_RUNNERS="${CONFIRM_LANE_MAX_REMOTE_RUNNERS:-6}"
POWEROFF_AFTER_RUN="${POWEROFF_AFTER_RUN:-true}"
VPS_BATCH_SESSION_ENABLE="${VPS_BATCH_SESSION_ENABLE:-1}"
CONFIRM_FASTLANE_UNIQUE_GROUPS="${CONFIRM_FASTLANE_UNIQUE_GROUPS:-1}"
CONFIRM_DISPATCH_ALLOW_WITH_DRIVER="${CONFIRM_DISPATCH_ALLOW_WITH_DRIVER:-0}"

mkdir -p "$STATE_DIR"

log() {
  printf '%s | %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*" >> "$LOG_FILE"
}

driver_loop_active() {
  if systemctl --user is-active --quiet autonomous-wfa-driver.service >/dev/null 2>&1; then
    return 0
  fi
  pgrep -f "scripts/optimization/autonomous_wfa_driver.sh" >/dev/null 2>&1
}

if [[ "$CONFIRM_DISPATCH_ALLOW_WITH_DRIVER" != "1" && "$CONFIRM_DISPATCH_ALLOW_WITH_DRIVER" != "true" ]]; then
  if driver_loop_active; then
    log "confirm_dispatch_agent_deferred reason=driver_active"
    exit 0
  fi
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "confirm_dispatch_agent_already_running"
  exit 0
fi
trap 'flock -u 9' EXIT

python3 - "$ROOT_DIR" "$FULLSPAN_DECISION_STATE_FILE" "$DECISION_NOTES_FILE" "$STATE_DIR" "$LOG_FILE" "$CONFIRM_LINEAGE_REGISTRY_FILE" "$FULLSPAN_CONFIRM_MIN_GROUPS" "$FULLSPAN_CONFIRM_MIN_REPLIES" "$CONFIRM_FASTLANE_LIMIT" "$CONFIRM_FASTLANE_PARALLEL" "$CONFIRM_FASTLANE_MAX_LIMIT" "$CONFIRM_FASTLANE_MAX_PARALLEL" "$CONFIRM_FASTLANE_COOLDOWN_SEC" "$SERVER_IP" "$SERVER_USER" "$POWEROFF_AFTER_RUN" "$CONFIRM_DISPATCHES_PER_CYCLE" "$CONFIRM_LANE_MAX_ACTIVE" "$CONFIRM_LANE_MAX_REMOTE_RUNNERS" "$CAPACITY_CONTROLLER_STATE_FILE" "$VPS_BATCH_SESSION_ENABLE" "$CONFIRM_FASTLANE_UNIQUE_GROUPS" <<'PY'
import csv
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
import fcntl
import sys
import shlex

root = Path(sys.argv[1])
state_path = Path(sys.argv[2])
notes_path = Path(sys.argv[3])
state_dir = Path(sys.argv[4])
log_path = Path(sys.argv[5])
registry_path = Path(sys.argv[6])
confirm_min_groups = int(float(sys.argv[7] or 2))
confirm_min_replies = int(float(sys.argv[8] or 2))
confirm_limit = int(float(sys.argv[9] or 1))
confirm_parallel = int(float(sys.argv[10] or 2))
confirm_max_limit = int(float(sys.argv[11] or 3))
confirm_max_parallel = int(float(sys.argv[12] or 4))
cooldown_sec = int(float(sys.argv[13] or 0))
server_ip = sys.argv[14]
server_user = sys.argv[15]
poweroff_after_run = sys.argv[16]
dispatches_per_cycle = int(float(sys.argv[17] or 2))
confirm_lane_max_active = int(float(sys.argv[18] or 2))
confirm_lane_max_remote_runners = int(float(sys.argv[19] or 6))
capacity_state_path = Path(sys.argv[20])
batch_session_enable_raw = str(sys.argv[21] or "0").strip().lower()
confirm_unique_groups_raw = str(sys.argv[22] or "1").strip().lower()
now_epoch = int(time.time())

batch_session_enable = batch_session_enable_raw in {"1", "true", "yes", "y", "on"}
confirm_unique_groups = confirm_unique_groups_raw in {"1", "true", "yes", "y", "on"}
if batch_session_enable:
    poweroff_after_run = "false"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{ts} | {msg}\n")


def load_state(path: Path):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_state(path: Path, state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_capacity_policy() -> dict:
    if not capacity_state_path.exists():
        return {}
    try:
        payload = json.loads(capacity_state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    policy = payload.get("policy", {})
    return policy if isinstance(policy, dict) else {}


def state_get(state: dict, queue: str, field: str, default=""):
    try:
        return state.get("queues", {}).get(queue, {}).get(field, default)
    except Exception:
        return default


def state_set_queue(state: dict, queue: str, updates: dict) -> None:
    queues = state.setdefault("queues", {})
    entry = queues.setdefault(queue, {})
    for key, value in updates.items():
        entry[key] = value


def metric_inc(state: dict, metric: str, delta: int = 1) -> None:
    metrics = state.setdefault("runtime_metrics", {})
    try:
        metrics[metric] = int(metrics.get(metric, 0)) + int(delta)
    except Exception:
        metrics[metric] = delta


def log_note(action: str, queue: str, reason: str, next_step: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": ts,
        "queue": queue,
        "action": action,
        "reason": reason,
        "next_step": next_step,
    }
    with notes_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def extract_lineage_uid(*values) -> str:
    queue = list(values)
    seen = set()
    while queue:
        current = queue.pop(0)
        ident = id(current)
        if ident in seen:
            continue
        seen.add(ident)
        if isinstance(current, dict):
            for key in ("lineage_uid", "candidate_uid"):
                value = str(current.get(key) or "").strip().lower()
                if value:
                    return value
            raw_meta = current.get("metadata_json")
            if isinstance(raw_meta, str):
                text = raw_meta.strip()
                if text.startswith("{") or text.startswith("["):
                    try:
                        queue.append(json.loads(text))
                    except Exception:
                        pass
            for value in current.values():
                if isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(current, list):
            queue.extend(current)
    return ""

def derive_candidate_uid(
    top_run_group: str,
    top_variant: str,
    top_score: str,
    existing: str = "",
    lineage_uid: str = "",
    top_metadata: str = "",
) -> str:
    current = str(existing or "").strip()
    if current:
        return current
    explicit_lineage = str(lineage_uid or "").strip().lower()
    if explicit_lineage:
        return explicit_lineage
    metadata_lineage = extract_lineage_uid(top_metadata)
    if metadata_lineage:
        return metadata_lineage
    evo_re = re.compile(r"\b(evo_[0-9a-f]{8,64})\b", re.IGNORECASE)
    for token in (top_variant, top_run_group):
        text = str(token or "").strip()
        if not text:
            continue
        m = evo_re.search(text)
        if m:
            return m.group(1).lower()
    seed = "|".join([str(top_run_group or "").strip().lower(), str(top_variant or "").strip().lower(), str(top_score or "").strip()])
    if not seed.strip("|"):
        return ""
    return "cand_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]

def register_lineage(
    queue_rel: str,
    source_queue_rel: str,
    candidate_uid: str,
    lineage_uid: str,
    top_run_group: str,
    top_variant: str,
    dispatch_id: str,
) -> bool:
    if not queue_rel or not source_queue_rel or not candidate_uid:
        return False
    cmd = [
        str(root / ".venv" / "bin" / "python"),
        str(root / "scripts/optimization/fullspan_lineage.py"),
        "register",
        "--registry",
        str(registry_path),
        "--queue-rel",
        str(queue_rel),
        "--source-queue-rel",
        str(source_queue_rel),
        "--candidate-uid",
        str(candidate_uid),
        "--lineage-uid",
        str(lineage_uid or ""),
        "--top-run-group",
        str(top_run_group or ""),
        "--top-variant",
        str(top_variant or ""),
        "--dispatch-id",
        str(dispatch_id or ""),
    ]
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, check=False)
    ok = proc.returncode == 0
    if not ok:
        log(
            "confirm_dispatch_agent lineage_register_failed queue={queue} source={source} uid={uid} rc={rc} stderr={stderr}".format(
                queue=queue_rel,
                source=source_queue_rel,
                uid=candidate_uid,
                rc=proc.returncode,
                stderr=(proc.stderr or "").strip()[-300:],
            )
        )
    return ok

def detect_remote_load() -> float:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=6",
        f"{server_user}@{server_ip}",
        "cat /proc/loadavg 2>/dev/null | awk '{print $1}'",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return -1.0
    try:
        return float((proc.stdout or "").strip().splitlines()[-1])
    except Exception:
        return -1.0

def detect_remote_runner_count() -> int:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=6",
        f"{server_user}@{server_ip}",
        "python3 - <<'PY'\n"
        "import os\n"
        "patterns=('watch_wfa_queue.sh','run_wfa_queue.py','run_wfa_fullcpu.sh','walk_forward')\n"
        "count=0\n"
        "for pid in os.listdir('/proc'):\n"
        "    if not pid.isdigit():\n"
        "        continue\n"
        "    try:\n"
        "        cmd=open(f'/proc/{pid}/cmdline','rb').read().replace(b'\\\\x00',b' ').decode('utf-8','ignore').strip()\n"
        "    except Exception:\n"
        "        continue\n"
        "    if not cmd or 'python3 - <<' in cmd or 'pgrep -f' in cmd:\n"
        "        continue\n"
        "    if any(p in cmd for p in patterns):\n"
        "        count += 1\n"
        "print(count)\n"
        "PY",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return -1
    try:
        return int(float((proc.stdout or "").strip().splitlines()[-1]))
    except Exception:
        return -1

def choose_confirm_params(entry: dict, confirm_count: int) -> tuple[int, int, dict]:
    limit = max(1, int(confirm_limit))
    parallel = max(1, int(confirm_parallel))

    max_limit = max(limit, int(confirm_max_limit))
    max_parallel = max(parallel, int(confirm_max_parallel))
    pending_since = int(float(entry.get("confirm_pending_since_epoch", 0) or 0))
    pending_age = max(0, now_epoch - pending_since) if pending_since > 0 else 0
    sla_confirm_sec = int(float(os.environ.get("SLA_CONFIRM_PENDING_SEC", "7200") or 7200))
    remote_load = detect_remote_load()
    remote_runners = detect_remote_runner_count()

    if confirm_count <= 0:
        limit += 1
    if pending_age >= sla_confirm_sec:
        limit += 1
        parallel += 1

    if remote_runners > 0:
        parallel = max(1, parallel - 1)
    elif remote_load >= 0:
        if remote_load <= 2.0:
            parallel += 1
        elif remote_load >= 8.0:
            parallel = max(1, parallel - 1)

    capacity_policy = load_capacity_policy()
    capacity_parallel_min = max(1, int(float(capacity_policy.get("confirm_parallel_min", 1) or 1)))
    capacity_parallel_max = max(capacity_parallel_min, int(float(capacity_policy.get("confirm_parallel_max", max_parallel) or max_parallel)))
    capacity_dispatches = max(1, int(float(capacity_policy.get("confirm_dispatches_per_cycle", dispatches_per_cycle) or dispatches_per_cycle)))
    capacity_lane_active = max(1, int(float(capacity_policy.get("confirm_lane_max_active", confirm_lane_max_active) or confirm_lane_max_active)))
    capacity_lane_remote = max(1, int(float(capacity_policy.get("confirm_lane_max_remote_runners", confirm_lane_max_remote_runners) or confirm_lane_max_remote_runners)))

    max_limit = max(max_limit, capacity_dispatches)
    parallel = max(capacity_parallel_min, parallel)
    max_parallel = min(max_parallel, capacity_parallel_max)
    if max_parallel < capacity_parallel_min:
        max_parallel = capacity_parallel_min

    limit = max(1, min(limit, max_limit))
    parallel = max(1, min(parallel, max_parallel))

    return limit, parallel, {
        "pending_age_sec": pending_age,
        "sla_confirm_sec": sla_confirm_sec,
        "remote_load1": remote_load,
        "remote_runner_count": remote_runners,
        "capacity_parallel_min": capacity_parallel_min,
        "capacity_parallel_max": capacity_parallel_max,
        "capacity_dispatches_per_cycle": capacity_dispatches,
        "capacity_lane_max_active": capacity_lane_active,
        "capacity_lane_max_remote_runners": capacity_lane_remote,
    }

def confirm_fastlane_pending(root_dir: Path, queue_rel: str) -> int:
    qpath = root_dir / queue_rel
    if not qpath.exists():
        return 0
    try:
        rows = list(csv.DictReader(qpath.open(newline="", encoding="utf-8")))
    except Exception:
        return 0
    pending = 0
    for r in rows:
        s = str(r.get("status") or "").strip().lower()
        if s in {"planned", "running", "stalled", "failed", "error"}:
            pending += 1
    return pending


def active_confirm_pending_count(state: dict) -> int:
    queues = state.get("queues", {}) if isinstance(state.get("queues"), dict) else {}
    total = 0
    for _queue, entry in queues.items():
        if not isinstance(entry, dict):
            continue
        qrel = str(entry.get("confirm_fastlane_queue_rel", "") or "").strip()
        if not qrel:
            continue
        total += 1 if confirm_fastlane_pending(root, qrel) > 0 else 0
    return total


def is_stress_shortlist_row(row: dict) -> bool:
    cfg = str(row.get("config_path") or "").strip()
    cfg_name = Path(cfg).name.lower()
    if cfg_name.endswith("_stress.yaml") or cfg_name.startswith("stress_"):
        return True

    for field in ("results_dir", "run_id", "run_name"):
        value = str(row.get(field) or "").strip().lower()
        if not value:
            continue
        if re.search(r"(^|[\/_-])stress([\/_-]|$)", value):
            return True
    return False


def shortlist_from_queue(queue_rel: str, top_variant: str, shortlist_path: Path, source_queue: Path) -> int:
    shortlist_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if source_queue.exists():
        try:
            rows = list(csv.DictReader(source_queue.open(newline="", encoding="utf-8")))
        except Exception:
            rows = []
    picked = []
    seen = set()
    top = (top_variant or "").strip().lower()

    for r in rows:
        cfg = str(r.get("config_path") or "").strip()
        if not cfg or cfg in seen:
            continue
        if is_stress_shortlist_row(r):
            continue
        blob = " ".join([cfg, str(r.get("results_dir") or ""), str(r.get("status") or "")]).lower()
        if top and top not in blob:
            continue
        seen.add(cfg)
        picked.append(
            {
                "config_path": cfg,
                "results_dir": str(r.get("results_dir") or ""),
                "status": "planned",
                "lineage_uid": extract_lineage_uid(r),
                "metadata_json": str(r.get("metadata_json") or ""),
            }
        )

    if not picked:
        for r in rows:
            cfg = str(r.get("config_path") or "").strip()
            if not cfg or cfg in seen:
                continue
            if is_stress_shortlist_row(r):
                continue
            seen.add(cfg)
            picked.append(
                {
                    "config_path": cfg,
                    "results_dir": str(r.get("results_dir") or ""),
                    "status": "planned",
                    "lineage_uid": extract_lineage_uid(r),
                    "metadata_json": str(r.get("metadata_json") or ""),
                }
            )
            break

    fieldnames = ["config_path", "results_dir", "status", "lineage_uid", "metadata_json"]
    with shortlist_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in picked[:1]:
            w.writerow(row)
    return len(picked[:1])


def shortlist_identity(shortlist_path: Path) -> tuple[str, str]:
    if not shortlist_path.exists():
        return "", ""
    try:
        rows = list(csv.DictReader(shortlist_path.open(newline="", encoding="utf-8")))
    except Exception:
        return "", ""
    if not rows:
        return "", ""
    row = rows[0] if isinstance(rows[0], dict) else {}
    return extract_lineage_uid(row), str(row.get("metadata_json") or "")


def build_and_dispatch(
    entry_queue: str,
    top_run_group: str,
    top_variant: str,
    top_score: str,
    candidate_uid: str,
    confirm_count: int,
    state: dict,
    entry_state: dict,
) -> bool:
    queue_basename = Path(entry_queue).parent.name
    safe_name = queue_basename.replace("/", "_").replace(".", "_")
    dispatch_seed = f"{candidate_uid}|{confirm_count}|{now_epoch}|{time.time_ns()}"
    dispatch_suffix = hashlib.sha1(dispatch_seed.encode("utf-8")).hexdigest()[:10]
    dispatch_id = f"d{now_epoch}_{dispatch_suffix}"
    queue_suffix = f"{safe_name}_{dispatch_id}" if confirm_unique_groups else safe_name

    shortlist_rel = Path("artifacts/wfa/aggregate/.autonomous") / f"confirm_fastlane_shortlist_{safe_name}.csv"
    confirm_queue_dir_rel = Path("artifacts/wfa/aggregate") / f"confirm_fastlane_{queue_suffix}"
    confirm_queue_rel = confirm_queue_dir_rel / "run_queue.csv"
    stress_dir_rel = Path("configs/confirm_fastlane") / queue_suffix / "stress"
    cycle_name = f"confirm_fastlane_{queue_suffix}"
    source_queue = root / entry_queue

    shortlist_path = state_dir / shortlist_rel
    shortlist_count = shortlist_from_queue(entry_queue, top_variant, shortlist_path, source_queue)
    if shortlist_count <= 0:
        log(f"confirm_dispatch_agent shortlist_empty queue={entry_queue}")
        metric_inc(state, "confirm_fastlane_trigger_empty_shortlist", 1)
        log_note(
            "CONFIRM_FASTLANE_SHORTLIST_EMPTY",
            entry_queue,
            "no_configs_matching_top_variant_or_shortlist",
            "wait_for_next_scan",
        )
        return False
    shortlist_lineage_uid, shortlist_metadata = shortlist_identity(shortlist_path)
    candidate_uid = derive_candidate_uid(
        top_run_group=top_run_group,
        top_variant=top_variant,
        top_score=top_score,
        existing=candidate_uid,
        lineage_uid=shortlist_lineage_uid,
        top_metadata=shortlist_metadata,
    )
    if not candidate_uid:
        return False

    limit_value, parallel_value, runtime_meta = choose_confirm_params(entry_state, confirm_count)

    build_cmd = [
        root / ".venv" / "bin" / "python",
        str(root / "scripts/optimization/build_confirm_queue.py"),
        "--shortlist-queue",
        str(shortlist_rel),
        "--cycle",
        cycle_name,
        "--queue-dir",
        str(confirm_queue_dir_rel),
        "--stress-config-dir",
        str(stress_dir_rel),
        "--limit",
        str(limit_value),
    ]
    qlog = state_dir / f"confirm_fastlane_{safe_name}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.log"
    try:
        with qlog.open("a", encoding="utf-8") as qf:
            subprocess.run([str(x) for x in build_cmd], cwd=str(root), stdout=qf, stderr=qf, check=True)
    except Exception as exc:
        log(f"confirm_dispatch_agent build_failed queue={entry_queue} err={type(exc).__name__}:{exc}")
        metric_inc(state, "confirm_fastlane_build_failed", 1)
        log_note(
            "CONFIRM_FASTLANE_BUILD_FAILED",
            entry_queue,
            f"build_confirm_queue_failed:{type(exc).__name__}:{exc}",
            "retry_later",
        )
        return False

    qlog.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(root / ".venv" / "bin" / "python"),
        str(root / "scripts/optimization/run_wfa_queue_powered.py"),
        "--queue",
        str(confirm_queue_rel),
        "--compute-host",
        server_ip,
        "--ssh-user",
        server_user,
        "--parallel",
        str(parallel_value),
        "--statuses",
        "planned",
        "--max-retries",
        "2",
        "--watchdog",
        "true",
        "--wait-completion",
        "false",
        "--postprocess",
        "true",
        "--poweroff",
        poweroff_after_run,
    ]
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    log_path = shlex.quote(str(qlog))
    root_path = shlex.quote(str(root))
    # run in background with same env + logging style as driver.
    full_cmd = f"cd {root_path} && AUTONOMOUS_MODE=1 ALLOW_HEAVY_RUN=1 {cmd_str} >>{log_path} 2>&1"
    try:
        subprocess.Popen(full_cmd, shell=True)
    except Exception as exc:
        log(f"confirm_dispatch_agent dispatch_failed queue={entry_queue} err={type(exc).__name__}:{exc}")
        metric_inc(state, "confirm_fastlane_dispatch_failed", 1)
        log_note(
            "CONFIRM_FASTLANE_DISPATCH_FAILED",
            entry_queue,
            f"dispatch_confirm_queue_failed:{type(exc).__name__}:{exc}",
            "retry_later",
        )
        return False

    state_set_queue(
        state,
        entry_queue,
        {
            "confirm_fastlane_queue_rel": str(confirm_queue_rel),
            "confirm_fastlane_last_trigger_epoch": now_epoch,
            "confirm_pending_since_epoch": now_epoch,
            "candidate_uid": candidate_uid,
            "lineage_uid": shortlist_lineage_uid or candidate_uid,
            "confirm_fastlane_last_dispatch_id": dispatch_id,
        },
    )

    lineage_ok = register_lineage(
        queue_rel=str(confirm_queue_rel),
        source_queue_rel=entry_queue,
        candidate_uid=candidate_uid,
        lineage_uid=shortlist_lineage_uid or candidate_uid,
        top_run_group=top_run_group,
        top_variant=top_variant,
        dispatch_id=dispatch_id,
    )

    metric_inc(state, "confirm_fastlane_trigger_count", 1)
    log(
        "confirm_dispatch_agent trigger queue={queue} confirm_queue={confirm_queue} top_variant={variant} candidate_uid={uid} dispatch_id={dispatch_id} confirm_count={count} limit={limit} parallel={parallel} load1={load} remote_runners={rr} lineage_ok={lineage}".format(
            queue=entry_queue,
            confirm_queue=confirm_queue_rel,
            variant=top_variant,
            uid=candidate_uid,
            dispatch_id=dispatch_id,
            count=confirm_count,
            limit=limit_value,
            parallel=parallel_value,
            load=runtime_meta.get("remote_load1"),
            rr=runtime_meta.get("remote_runner_count"),
            lineage=int(bool(lineage_ok)),
        )
    )
    log_note(
        "CONFIRM_FASTLANE_TRIGGER",
        entry_queue,
        f"confirm_fastlane_queue={confirm_queue_rel} top_variant={top_variant} candidate_uid={candidate_uid} dispatch_id={dispatch_id} limit={limit_value} parallel={parallel_value}",
        "await_confirm_replay",
    )
    return True


def safe_lock_name(lock_file: Path):
    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def release_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except Exception:
        pass

state = load_state(state_path)
runtime_metrics = state.setdefault("runtime_metrics", {})
runtime_metrics["confirm_fastlane_cycle_dispatch_count"] = 0

queues = []
raw_queues = state.get("queues", {}) if isinstance(state.get("queues", {}), dict) else {}
for queue, entry in raw_queues.items():
    if not isinstance(entry, dict):
        continue
    try:
        strict_pass_count = int(float(entry.get("strict_pass_count", 0) or 0))
        strict_run_groups = int(float(entry.get("strict_run_group_count", 0) or 0))
        confirm_count = int(float(entry.get("confirm_count", 0) or 0))
    except Exception:
        continue

    verdict = str(entry.get("promotion_verdict", "")).strip().upper()
    top_run_group = str(entry.get("top_run_group", "")).strip()
    top_variant = str(entry.get("top_variant", "")).strip()
    top_score = str(entry.get("top_score", "")).strip()
    candidate_uid = derive_candidate_uid(
        top_run_group=top_run_group,
        top_variant=top_variant,
        top_score=top_score,
        existing=str(entry.get("candidate_uid", "")).strip(),
        lineage_uid=str(entry.get("lineage_uid", "")).strip(),
        top_metadata=str(entry.get("top_metadata", "")).strip(),
    )

    if not top_variant:
        continue
    if not candidate_uid:
        continue
    if strict_pass_count <= 0:
        continue
    if strict_run_groups < confirm_min_groups:
        continue
    if confirm_count >= confirm_min_replies:
        continue
    if verdict not in ("PROMOTE_PENDING_CONFIRM", "PROMOTE_DEFER_CONFIRM"):
        continue

    source_queue = root / queue
    if not source_queue.exists():
        continue

    state_ts = entry.get("last_update")
    queues.append(
        (
            state_ts or "",
            queue,
            strict_pass_count,
            strict_run_groups,
            confirm_count,
            top_run_group,
            top_variant,
            top_score,
            candidate_uid,
            entry,
        )
    )

if not queues:
    metric_inc(state, "confirm_fastlane_scan_noop", 1)
    save_state(state_path, state)
    sys.exit(0)

queues.sort(key=lambda item: str(item[0]))

for state_ts, queue_rel, _strict_pass_count, _strict_run_groups, confirm_count, top_run_group, top_variant, top_score, candidate_uid, entry_state in queues:
    capacity_policy = load_capacity_policy()
    effective_dispatches_per_cycle = max(1, int(float(capacity_policy.get("confirm_dispatches_per_cycle", dispatches_per_cycle) or dispatches_per_cycle)))
    effective_lane_max_active = max(1, int(float(capacity_policy.get("confirm_lane_max_active", confirm_lane_max_active) or confirm_lane_max_active)))
    effective_lane_max_remote = max(1, int(float(capacity_policy.get("confirm_lane_max_remote_runners", confirm_lane_max_remote_runners) or confirm_lane_max_remote_runners)))

    if effective_dispatches_per_cycle > 0:
        try:
            already = int(state.setdefault("runtime_metrics", {}).get("confirm_fastlane_cycle_dispatch_count", 0))
        except Exception:
            already = 0
        if already >= effective_dispatches_per_cycle:
            metric_inc(state, "confirm_fastlane_skip_cycle_quota", 1)
            break

    active_pending_now = active_confirm_pending_count(state)
    if effective_lane_max_active > 0 and active_pending_now >= effective_lane_max_active:
        metric_inc(state, "confirm_fastlane_skip_lane_active_cap", 1)
        continue

    remote_runners_now = detect_remote_runner_count()
    if effective_lane_max_remote > 0 and remote_runners_now >= effective_lane_max_remote:
        metric_inc(state, "confirm_fastlane_skip_remote_busy", 1)
        continue

    last_trigger = int(float(state_get(state, queue_rel, "confirm_fastlane_last_trigger_epoch", 0) or 0))
    if now_epoch - last_trigger < cooldown_sec:
        metric_inc(state, "confirm_fastlane_skip_cooldown", 1)
        continue

    existing_queue = str(state_get(state, queue_rel, "confirm_fastlane_queue_rel", "") or "").strip()
    if existing_queue:
        existing_pending = confirm_fastlane_pending(root, existing_queue)
        if existing_pending > 0:
            metric_inc(state, "confirm_fastlane_skip_existing_pending", 1)
            continue

    safe_name = Path(queue_rel).parent.name.replace("/", "_").replace(".", "_")
    lock_target = state_dir / f".{safe_name}.confirm_dispatch.lock"
    lock_fd = safe_lock_name(lock_target)
    if lock_fd is None:
        metric_inc(state, "confirm_fastlane_skip_lock", 1)
        continue

    try:
        metric_inc(state, "confirm_fastlane_trigger_attempt", 1)
        dispatched = build_and_dispatch(
            queue_rel,
            top_run_group,
            top_variant,
            top_score,
            candidate_uid,
            confirm_count,
            state,
            entry_state,
        )
        if dispatched:
            metrics = state.setdefault("runtime_metrics", {})
            try:
                metrics["confirm_fastlane_cycle_dispatch_count"] = int(metrics.get("confirm_fastlane_cycle_dispatch_count", 0)) + 1
            except Exception:
                metrics["confirm_fastlane_cycle_dispatch_count"] = 1
    finally:
        release_lock(lock_fd)

save_state(state_path, state)
PY
