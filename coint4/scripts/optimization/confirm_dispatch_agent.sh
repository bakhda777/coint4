#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
STATE_DIR="$ROOT_DIR/artifacts/wfa/aggregate/.autonomous"
LOG_FILE="$STATE_DIR/confirm_dispatch_agent.log"
LOCK_FILE="$STATE_DIR/confirm_dispatch_agent.lock"
FULLSPAN_DECISION_STATE_FILE="$STATE_DIR/fullspan_decision_state.json"
DECISION_NOTES_FILE="$STATE_DIR/decision_notes.jsonl"
SERVER_IP="${SERVER_IP:-85.198.90.128}"
SERVER_USER="${SERVER_USER:-root}"
FULLSPAN_CONFIRM_MIN_GROUPS="${FULLSPAN_CONFIRM_MIN_GROUPS:-2}"
FULLSPAN_CONFIRM_MIN_REPLIES="${FULLSPAN_CONFIRM_MIN_REPLIES:-2}"
CONFIRM_FASTLANE_LIMIT="${CONFIRM_FASTLANE_LIMIT:-1}"
CONFIRM_FASTLANE_PARALLEL="${CONFIRM_FASTLANE_PARALLEL:-2}"
CONFIRM_FASTLANE_COOLDOWN_SEC="${CONFIRM_FASTLANE_COOLDOWN_SEC:-1800}"
POWEROFF_AFTER_RUN="${POWEROFF_AFTER_RUN:-true}"

mkdir -p "$STATE_DIR"

log() {
  printf '%s | %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*" >> "$LOG_FILE"
}

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "confirm_dispatch_agent_already_running"
  exit 0
fi
trap 'flock -u 9' EXIT

python3 - "$ROOT_DIR" "$FULLSPAN_DECISION_STATE_FILE" "$DECISION_NOTES_FILE" "$STATE_DIR" "$LOG_FILE" "$FULLSPAN_CONFIRM_MIN_GROUPS" "$FULLSPAN_CONFIRM_MIN_REPLIES" "$CONFIRM_FASTLANE_LIMIT" "$CONFIRM_FASTLANE_PARALLEL" "$CONFIRM_FASTLANE_COOLDOWN_SEC" "$SERVER_IP" "$SERVER_USER" "$POWEROFF_AFTER_RUN" <<'PY'
import csv
import json
import os
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
confirm_min_groups = int(float(sys.argv[6] or 2))
confirm_min_replies = int(float(sys.argv[7] or 2))
confirm_limit = sys.argv[8]
confirm_parallel = sys.argv[9]
cooldown_sec = int(float(sys.argv[10] or 0))
server_ip = sys.argv[11]
server_user = sys.argv[12]
poweroff_after_run = sys.argv[13]
now_epoch = int(time.time())


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
        if cfg.endswith("_stress.yaml"):
            continue
        blob = " ".join([cfg, str(r.get("results_dir") or ""), str(r.get("status") or "")]).lower()
        if top and top not in blob:
            continue
        seen.add(cfg)
        picked.append({"config_path": cfg, "results_dir": str(r.get("results_dir") or ""), "status": "planned"})

    if not picked:
        for r in rows:
            cfg = str(r.get("config_path") or "").strip()
            if not cfg or cfg in seen:
                continue
            if cfg.endswith("_stress.yaml"):
                continue
            seen.add(cfg)
            picked.append({"config_path": cfg, "results_dir": str(r.get("results_dir") or ""), "status": "planned"})
            break

    with shortlist_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config_path", "results_dir", "status"])
        w.writeheader()
        for row in picked[:1]:
            w.writerow(row)
    return len(picked[:1])


def build_and_dispatch(entry_queue: str, top_variant: str, confirm_count: int, state: dict) -> None:
    queue_basename = Path(entry_queue).parent.name
    safe_name = queue_basename.replace("/", "_").replace(".", "_")

    shortlist_rel = Path("artifacts/wfa/aggregate/.autonomous") / f"confirm_fastlane_shortlist_{safe_name}.csv"
    confirm_queue_dir_rel = Path("artifacts/wfa/aggregate") / f"confirm_fastlane_{safe_name}"
    confirm_queue_rel = confirm_queue_dir_rel / "run_queue.csv"
    stress_dir_rel = Path("configs/confirm_fastlane") / safe_name / "stress"
    cycle_name = f"confirm_fastlane_{safe_name}"
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
        return

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
        str(confirm_limit),
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
        return

    state_set_queue(
        state,
        entry_queue,
        {
            "confirm_fastlane_queue_rel": str(confirm_queue_rel),
            "confirm_fastlane_last_trigger_epoch": now_epoch,
            "confirm_pending_since_epoch": now_epoch,
        },
    )

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
        str(confirm_parallel),
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
    subprocess.Popen(full_cmd, shell=True)

    metric_inc(state, "confirm_fastlane_trigger_count", 1)
    log(f"confirm_dispatch_agent trigger queue={entry_queue} confirm_queue={confirm_queue_rel} top_variant={top_variant} confirm_count={confirm_count}")
    log_note(
        "CONFIRM_FASTLANE_TRIGGER",
        entry_queue,
        f"confirm_fastlane_queue={confirm_queue_rel} top_variant={top_variant}",
        "await_confirm_replay",
    )


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
    top_variant = str(entry.get("top_variant", "")).strip()

    if not top_variant:
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
    queues.append((state_ts or "", queue, strict_pass_count, strict_run_groups, confirm_count, top_variant))

if not queues:
    metric_inc(state, "confirm_fastlane_scan_noop", 1)
    save_state(state_path, state)
    sys.exit(0)

queues.sort(key=lambda item: str(item[0]))

for state_ts, queue_rel, _strict_pass_count, _strict_run_groups, confirm_count, top_variant in queues:
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
        build_and_dispatch(queue_rel, top_variant, confirm_count, state)
    finally:
        release_lock(lock_fd)

save_state(state_path, state)
PY
