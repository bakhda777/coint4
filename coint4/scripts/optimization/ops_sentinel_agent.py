#!/usr/bin/env python3
"""Ops sentinel agent for WFA operations.

Produces:
- machine-readable JSONL events at artifacts/wfa/aggregate/.autonomous/ops_sentinel_events.jsonl
- concise human alerts in event.payload['human'] (for report consumption)
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def now_ts() -> int:
    return int(time.time())


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def dump_json(path: Path, data) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def collect_queue_stats(root: Path) -> Dict[str, Dict[str, int]]:
    """Collect queue metrics keyed by repo-relative queue path."""
    queues: Dict[str, Dict[str, int]] = {}
    agg_root = root / "artifacts" / "wfa" / "aggregate"
    for q in sorted(agg_root.rglob("run_queue.csv")):
        rel = str(q.relative_to(root)).replace("\\", "/")
        counts = defaultdict(int)
        try:
            with q.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    status = (row.get("status") or "").strip().lower()
                    counts["total"] += 1
                    if status in {"planned", "running", "stalled", "failed", "error", "active"}:
                        counts["pending"] += 1
                    if status == "running":
                        counts["running"] += 1
                    elif status == "stalled":
                        counts["stalled"] += 1
                    elif status == "failed" or status == "error":
                        counts["failed"] += 1
                    elif status == "completed":
                        counts["completed"] += 1
        except Exception:
            continue

        queues[rel] = {
            "queue": rel,
            "pending": int(counts.get("pending", 0)),
            "running": int(counts.get("running", 0)),
            "stalled": int(counts.get("stalled", 0)),
            "failed": int(counts.get("failed", 0)),
            "completed": int(counts.get("completed", 0)),
            "total": int(counts.get("total", 0)),
        }
    return queues


def detect_remote_queue_runs(server_user: str, server_ip: str, ssh_key: str) -> List[str]:
    """Return list of remote queue paths from running processes."""
    remote_cmd = (
        "ps -ef | egrep 'run_wfa_queue.py|watch_wfa_queue.sh|run_wfa_fullcpu.sh|run_fullspan_decision_cycle.py' | "+
        "egrep -v 'egrep|grep'"
    )
    p = subprocess.run(
        [
            "ssh",
            "-i",
            ssh_key,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=6",
            f"{server_user}@{server_ip}",
            remote_cmd,
        ],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        return []

    queues: List[str] = []
    seen = set()
    for line in p.stdout.splitlines():
        try:
            tokens = shlex.split(line)
        except Exception:
            continue

        q = None
        for idx, tok in enumerate(tokens):
            if tok == "--queue" and idx + 1 < len(tokens):
                q = tokens[idx + 1]
                break
        if q is None:
            for tok in tokens:
                if "run_queue.csv" in tok and "artifacts/wfa/aggregate" in tok:
                    q = tok
                    break
        if not q:
            continue

        if q.startswith("/opt/coint4/"):
            q = q[len("/opt/"):]
        if q.startswith("/home/") and "/coint4/" in q:
            q = q.split("/coint4/", 1)[1]
            if q and not q.startswith("artifacts/"):
                q = "coint4/" + q
        if q.startswith("coint4/"):
            q = q

        marker = "artifacts/wfa/aggregate/"
        if marker in q:
            q = q[q.find(marker) :]
        if not q.endswith("run_queue.csv"):
            continue
        if q not in seen:
            seen.add(q)
            queues.append(q)

    return queues


def detect_remote_runner_active(server_user: str, server_ip: str, ssh_key: str) -> bool:
    p = subprocess.run(
        [
            "ssh",
            "-i",
            ssh_key,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=6",
            f"{server_user}@{server_ip}",
            "pgrep -f 'run_wfa_queue.py|watch_wfa_queue.sh|run_wfa_fullcpu.sh' >/dev/null 2>&1; echo $?",
        ],
        capture_output=True,
        text=True,
    )
    rc_txt = p.stdout.strip().splitlines()[-1] if p.stdout else "1"
    try:
        return rc_txt == "0"
    except Exception:
        return False


def force_sync_queue(root: Path, queue_rel: str, log_file: Path, dry_run: bool) -> bool:
    script = root / "scripts" / "optimization" / "sync_queue_status.py"
    if dry_run:
        return False
    py = root / ".venv" / "bin" / "python"
    python = str(py) if py.exists() else "python3"
    cmd = [python, str(script), "--queue", queue_rel]
    p = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=90,
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{now_iso()}] sync_queue_status {queue_rel} rc={p.returncode}\n")
        if p.stdout:
            f.write(p.stdout)
        if p.stderr:
            f.write(p.stderr)
    return p.returncode == 0


def load_fullspan_state(root: Path) -> Dict:
    path = root / "artifacts" / "wfa" / "aggregate" / ".autonomous" / "fullspan_decision_state.json"
    data = load_json(path, {})
    return data if isinstance(data, dict) else {}


def emit_event(args, state: Dict, queue: str, event: str, severity: str, payload: Dict, human: str) -> None:
    ts = now_iso()
    rec = {
        "ts": ts,
        "event": event,
        "severity": severity,
        "queue": queue,
        "payload": payload,
        "human": human,
    }
    append_jsonl(args.events_file, rec)
    with args.log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {event} queue={queue} {human}\n")


def can_emit(state: Dict, key: str, now_i: int, ttl_sec: int = 300) -> bool:
    last = int(float(state.get("event_last_emit", {}).get(key, 0) or 0))
    if now_i - last < ttl_sec:
        return False
    state.setdefault("event_last_emit", {})[key] = now_i
    return True


def enforce_poweroff(root: Path, server_user: str, server_ip: str, ssh_key: str, dry_run: bool) -> bool:
    if dry_run:
        return False
    # check host reachability
    ping = subprocess.run(
        [
            "ssh",
            "-i",
            ssh_key,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=6",
            f"{server_user}@{server_ip}",
            "echo",
            "ok",
        ],
        capture_output=True,
        text=True,
    )
    if ping.returncode != 0:
        return False

    shutdown = subprocess.run(
        [
            "ssh",
            "-i",
            ssh_key,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=6",
            f"{server_user}@{server_ip}",
            "shutdown",
            "-h",
            "now",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    with (root / "artifacts" / "wfa" / "aggregate" / ".autonomous" / "ops_sentinel.log").open(
        "a", encoding="utf-8"
    ) as f:
        f.write(f"[{now_iso()}] poweroff rc={shutdown.returncode}\n")
    return shutdown.returncode == 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None, help="Project root")
    p.add_argument("--dry-run", action="store_true", help="Do not execute side-effects")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    root = (Path(str(args.root).replace(',', '').strip()) if args.root else Path(__file__).resolve().parents[2]).resolve()
    state_dir = root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    args.events_file = state_dir / "ops_sentinel_events.jsonl"
    args.log_file = state_dir / "ops_sentinel.log"
    state_file = state_dir / "ops_sentinel_state.json"

    cfg = {
        "no_runner_ttl_sec": int(os.environ.get("OPS_SENTINEL_NO_RUNNER_TTL_SEC", "900")),
        "confirm_pending_ttl_sec": int(os.environ.get("OPS_SENTINEL_CONFIRM_PENDING_TTL_SEC", os.environ.get("SLA_CONFIRM_PENDING_SEC", "7200"))),
        "first_strict_pass_ttl_sec": int(os.environ.get("OPS_SENTINEL_FIRST_STRICT_PASS_TTL_SEC", "3600")),
        "idle_poweroff_cooldown_sec": int(os.environ.get("OPS_SENTINEL_IDLE_POWEROFF_COOLDOWN_SEC", "1200")),
        "idle_poweroff_repeat_sec": int(os.environ.get("OPS_SENTINEL_IDLE_POWEROFF_REPEAT_SEC", "3600")),
        "server_user": os.environ.get("SERVER_USER", "root"),
        "server_ip": os.environ.get("SERVER_IP", "85.198.90.128"),
        "ssh_key": os.environ.get("SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519")),
        "dry_run": bool(args.dry_run),
    }

    if not cfg["server_ip"]:
        cfg["server_ip"] = "85.198.90.128"

    defaults = {
        "state_version": 1,
        "no_runner_since": {},
        "confirm_overdue_since": {},
        "first_strict_pass_started": 0,
        "no_work_since": 0,
        "last_poweroff_epoch": 0,
        "event_last_emit": {},
        "runtime": {},
    }
    state = load_json(state_file, defaults)
    if not isinstance(state, dict):
        state = dict(defaults)

    for k, v in defaults.items():
        state.setdefault(k, v)

    t0 = now_ts()
    queues = collect_queue_stats(root)
    total_pending = sum(v.get("pending", 0) for v in queues.values())
    total_running = sum(v.get("running", 0) for v in queues.values())

    remote_queues = []
    remote_running = 0
    try:
        remote_queues = detect_remote_queue_runs(cfg["server_user"], cfg["server_ip"], cfg["ssh_key"])
        remote_running = int(detect_remote_runner_active(cfg["server_user"], cfg["server_ip"], cfg["ssh_key"]))
    except Exception:
        remote_queues = []
        remote_running = 0

    remote_set = set(remote_queues)

    # Remote-sync sentinel: force sync queue statuses when remote runner(s) are active.
    if remote_queues:
        for q in remote_queues:
            if q in queues:
                sync_ok = force_sync_queue(root, q, args.log_file, cfg["dry_run"])
                if can_emit(state, f"remote_sync:{q}", t0, ttl_sec=120):
                    emit_event(
                        args,
                        state,
                        q,
                        "OPS_SENTINEL_REMOTE_SYNC_FORCE",
                        "info",
                        {"queue": q, "sync_ok": bool(sync_ok), "remote_running": bool(remote_running)},
                        f"remote runner active → force sync {q}",
                    )
            elif can_emit(state, f"remote_sync_unknown:{q}", t0, ttl_sec=120):
                emit_event(
                    args,
                    state,
                    q,
                    "OPS_SENTINEL_REMOTE_SYNC_UNKNOWN",
                    "warning",
                    {"queue": q},
                    f"active remote queue not tracked locally: {q}",
                )

    # pending>0 with no runner for TTL
    for q, st in queues.items():
        if st.get("pending", 0) <= 0:
            state["no_runner_since"].pop(q, None)
            continue
        if st.get("running", 0) > 0:
            state["no_runner_since"].pop(q, None)
            continue
        if q in remote_set:
            state["no_runner_since"].pop(q, None)
            continue

        since = int(float(state["no_runner_since"].get(q, 0) or 0))
        if since <= 0:
            state["no_runner_since"][q] = t0
            continue

        age = t0 - since
        if age >= cfg["no_runner_ttl_sec"] and can_emit(state, f"no_runner:{q}", t0, ttl_sec=1800):
            emit_event(
                args,
                state,
                q,
                "OPS_SENTINEL_NO_RUNNER_PENDING",
                "warning",
                {
                    "pending": st["pending"],
                    "running": 0,
                    "remote_running": 0,
                    "ttl_sec": cfg["no_runner_ttl_sec"],
                    "age_sec": age,
                },
                f"pending={st['pending']} без раннеров уже {age}s",
            )

    # confirm pending overdue from fullspan decision state.
    fullspan = load_fullspan_state(root)
    runtime = fullspan.get("runtime_metrics", {}) if isinstance(fullspan, dict) else {}
    state["runtime"] = runtime if isinstance(runtime, dict) else {}

    queues_state = fullspan.get("queues", {}) if isinstance(fullspan, dict) else {}
    for q, st in (queues_state or {}).items():
        if not isinstance(st, dict):
            continue
        verdict = str(st.get("promotion_verdict", "")).strip().upper()
        if verdict not in {"PROMOTE_PENDING_CONFIRM", "PROMOTE_DEFER_CONFIRM"}:
            state["confirm_overdue_since"].pop(q, None)
            continue

        pending_since = int(float(st.get("confirm_pending_since_epoch", 0) or 0))
        if pending_since <= 0:
            state["confirm_overdue_since"].pop(q, None)
            continue

        age = t0 - pending_since
        if age < cfg["confirm_pending_ttl_sec"]:
            state["confirm_overdue_since"][q] = pending_since
            continue

        last = int(float(state["confirm_overdue_since"].get(q, 0) or 0))
        if not can_emit(state, f"confirm_overdue:{q}", t0, ttl_sec=1800):
            continue

        emit_event(
            args,
            state,
            q,
            "OPS_SENTINEL_CONFIRM_PENDING_OVERDUE",
            "warning",
            {
                "pending_confirmation_since": pending_since,
                "age_sec": age,
                "sla_sec": cfg["confirm_pending_ttl_sec"],
                "promotion_verdict": verdict,
                "confirm_count": int(float(st.get("confirm_count", 0) or 0)),
                "strict_pass_count": int(float(st.get("strict_pass_count", 0) or 0)),
            },
            f"confirm pending > {cfg['confirm_pending_ttl_sec']}s ({q})",
        )
        state["confirm_overdue_since"][q] = t0

    for q in list(state["confirm_overdue_since"].keys()):
        if q not in queues_state or not isinstance(queues_state.get(q), dict):
            state["confirm_overdue_since"][q] = 0

    # SLA: time_to_first_strict_pass breach.
    strict_fullspan_pass_count = int(float(runtime.get("strict_fullspan_pass_count", 0) or 0)) if isinstance(runtime, dict) else 0
    last_strict_pass_epoch = int(float(runtime.get("last_strict_pass_epoch", 0) or 0)) if isinstance(runtime, dict) else 0
    if strict_fullspan_pass_count <= 0 and last_strict_pass_epoch <= 0:
        if int(float(state.get("first_strict_pass_started", 0) or 0)) <= 0:
            state["first_strict_pass_started"] = t0
        elif can_emit(state, "first_strict_pass", t0, ttl_sec=cfg["first_strict_pass_ttl_sec"] + 1):
            age = t0 - int(state["first_strict_pass_started"])
            if age >= cfg["first_strict_pass_ttl_sec"]:
                emit_event(
                    args,
                    state,
                    "global",
                    "OPS_SENTINEL_FIRST_STRICT_PASS_BREACH",
                    "critical",
                    {
                        "strict_fullspan_pass_count": strict_fullspan_pass_count,
                        "age_sec": age,
                        "sla_sec": cfg["first_strict_pass_ttl_sec"],
                    },
                    f"time_to_first_strict_pass > {cfg['first_strict_pass_ttl_sec']}s",
                )
    else:
        state["first_strict_pass_started"] = 0

    # Idle cost controller.
    if total_pending <= 0 and remote_running <= 0:
        if int(float(state.get("no_work_since", 0) or 0)) <= 0:
            state["no_work_since"] = t0
        else:
            no_work_for = t0 - int(state["no_work_since"])
            if no_work_for >= cfg["idle_poweroff_cooldown_sec"] and can_emit(
                state,
                "idle_poweroff",
                t0,
                ttl_sec=cfg["idle_poweroff_repeat_sec"],
            ):
                if (t0 - int(state.get("last_poweroff_epoch", 0) or 0)) >= cfg["idle_poweroff_repeat_sec"]:
                    ok = enforce_poweroff(root, cfg["server_user"], cfg["server_ip"], cfg["ssh_key"], cfg["dry_run"])
                    if ok:
                        state["last_poweroff_epoch"] = t0
                        emit_event(
                            args,
                            state,
                            "global",
                            "OPS_SENTINEL_IDLE_POWEROFF",
                            "warning",
                            {"cooldown_sec": cfg["idle_poweroff_cooldown_sec"], "pending": total_pending},
                            "no pending/runners > cooldown; VPS отключен",
                        )
                    else:
                        emit_event(
                            args,
                            state,
                            "global",
                            "OPS_SENTINEL_IDLE_POWEROFF_SKIP",
                            "warning",
                            {"reason": "vps_unreachable_or_shutdown_failed", "cooldown_sec": cfg["idle_poweroff_cooldown_sec"]},
                            "VPS не остановлен: unreachable/command failed",
                        )
    else:
        state["no_work_since"] = 0

    if not cfg["dry_run"]:
        dump_json(state_file, state)

    elapsed_ms = (now_ts() - t0) * 1000
    with args.log_file.open("a", encoding="utf-8") as f:
        f.write(
            f"[{now_iso()}] completed queues={len(queues)} pending={total_pending} running={total_running} "
            f"remote_running={remote_running} events={len(state.get('event_last_emit', {}))}\n"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
