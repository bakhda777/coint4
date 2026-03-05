#!/usr/bin/env python3
"""Progress guard for autonomous WFA loop.

Checks every few minutes:
- local driver process is alive while there is pending work
- driver log heartbeat freshness
- VPS SSH reachability

Applies safe remediation:
- start/restart driver with cooldown and restart rate limits
- emit machine-readable events and human log lines
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_epoch() -> int:
    return int(time.time())


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{now_iso()}] {message}\n")


def parse_log_ts(line: str) -> int:
    # line format: 2026-03-05T11:31:52Z | ...
    if len(line) < 20:
        return 0
    ts = line[:20]
    try:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return 0
    return int(dt.timestamp())


def last_driver_log_epoch(path: Path, max_lines: int = 500) -> int:
    if not path.exists():
        return 0
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return 0
    for line in reversed(lines[-max_lines:]):
        ts = parse_log_ts(line)
        if ts > 0:
            return ts
    return 0


def count_pending(aggregate_root: Path) -> int:
    pending_status = {"planned", "running", "stalled", "failed", "error", "active"}
    pending = 0
    for queue in aggregate_root.rglob("run_queue.csv"):
        raw = str(queue).replace("\\", "/")
        if "/rollup/" in raw or "/.autonomous/" in raw:
            continue
        try:
            with queue.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    status = str(row.get("status") or "").strip().lower()
                    if status in pending_status:
                        pending += 1
        except Exception:
            continue
    return pending


def _cmdline_parts(pid: str) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except Exception:
        return []
    if not raw:
        return []
    return [p.decode("utf-8", "ignore") for p in raw.split(b"\x00") if p]


def find_driver_pids(driver_script: Path) -> list[int]:
    out: list[int] = []
    self_pid = os.getpid()
    parent_pid = os.getppid()
    needle = str(driver_script.resolve())
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        ipid = int(pid)
        if ipid in {self_pid, parent_pid}:
            continue
        parts = _cmdline_parts(pid)
        if not parts:
            continue
        if needle in parts:
            out.append(ipid)
    return sorted(set(out))


def count_local_queue_runners() -> int:
    patterns = (
        "run_wfa_queue_powered.py",
        "run_wfa_queue.py",
        "watch_wfa_queue.sh",
        "recover_stalled_queue.sh",
    )
    count = 0
    self_pid = os.getpid()
    parent_pid = os.getppid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        ipid = int(pid)
        if ipid in {self_pid, parent_pid}:
            continue
        parts = _cmdline_parts(pid)
        if not parts:
            continue
        cmd = " ".join(parts)
        if "python3 - <<" in cmd:
            continue
        if any(p in cmd for p in patterns):
            count += 1
    return count


def ssh_reachable(server_user: str, server_ip: str) -> bool:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=6",
        f"{server_user}@{server_ip}",
        "echo",
        "ok",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode == 0


def _py_executable(root: Path) -> str:
    venv_py = root / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return "python3"


def _run_capture(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()


def _serverspace_find(root: Path, server_ip: str) -> dict[str, str]:
    api_script = root / "scripts" / "vps" / "serverspace_api.py"
    if not api_script.exists():
        return {}
    rc, out, _err = _run_capture([_py_executable(root), str(api_script), "--ip", server_ip, "find"])
    if rc != 0 or not out:
        return {}
    try:
        data = json.loads(out.splitlines()[-1])
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        "server_id": str(data.get("server_id") or data.get("id") or "").strip(),
        "state": str(data.get("state") or "").strip(),
        "name": str(data.get("name") or "").strip(),
    }


def can_power_cycle(state: dict[str, Any], now_i: int, cooldown_sec: int, max_per_hour: int) -> tuple[bool, str]:
    items = state.get("power_cycle_epochs")
    if not isinstance(items, list):
        items = []
    state["power_cycle_epochs"] = [int(x) for x in items if isinstance(x, (int, float)) and now_i - int(x) <= 3600]
    last_cycle = int(float(state.get("last_power_cycle_epoch", 0) or 0))
    if cooldown_sec > 0 and now_i - last_cycle < cooldown_sec:
        return False, "cooldown"
    if max_per_hour > 0 and len(state.get("power_cycle_epochs", [])) >= max_per_hour:
        return False, "hourly_limit"
    return True, "ok"


def power_cycle_vps(
    root: Path,
    *,
    server_user: str,
    server_ip: str,
    shutdown_wait_sec: int,
    boot_wait_sec: int,
    dry_run: bool,
) -> tuple[bool, str]:
    server_id = str((_serverspace_find(root, server_ip).get("server_id") or "")).strip()
    if not server_id:
        return False, "server_id_unresolved"
    if dry_run:
        return True, f"dry_run_server_id={server_id}"

    api_script = root / "scripts" / "vps" / "serverspace_api.py"
    py = _py_executable(root)
    rc, _out, err = _run_capture([py, str(api_script), "shutdown", "--id", server_id])
    if rc != 0:
        return False, f"shutdown_failed:{err[:180]}"
    time.sleep(max(5, shutdown_wait_sec))

    rc, _out, err = _run_capture([py, str(api_script), "power-on", "--id", server_id])
    if rc != 0:
        return False, f"power_on_failed:{err[:180]}"

    deadline = now_epoch() + max(60, boot_wait_sec)
    while now_epoch() < deadline:
        if ssh_reachable(server_user, server_ip):
            return True, f"recovered_server_id={server_id}"
        time.sleep(5)
    return False, "ssh_not_ready_after_power_cycle"


def should_emit(state: dict[str, Any], key: str, now_i: int, ttl_sec: int) -> bool:
    event_last_emit = state.setdefault("event_last_emit", {})
    last = int(float(event_last_emit.get(key, 0) or 0))
    if now_i - last < ttl_sec:
        return False
    event_last_emit[key] = now_i
    return True


def prune_restart_epochs(state: dict[str, Any], now_i: int, window_sec: int = 3600) -> None:
    items = state.get("restart_epochs")
    if not isinstance(items, list):
        items = []
    state["restart_epochs"] = [int(x) for x in items if isinstance(x, (int, float)) and now_i - int(x) <= window_sec]


def can_restart(state: dict[str, Any], now_i: int, cooldown_sec: int, max_per_hour: int) -> tuple[bool, str]:
    prune_restart_epochs(state, now_i)
    last_restart = int(float(state.get("last_restart_epoch", 0) or 0))
    if cooldown_sec > 0 and now_i - last_restart < cooldown_sec:
        return False, "cooldown"
    if max_per_hour > 0 and len(state.get("restart_epochs", [])) >= max_per_hour:
        return False, "hourly_limit"
    return True, "ok"


def start_driver(root: Path, driver_script: Path, dry_run: bool) -> bool:
    if dry_run:
        return True
    try:
        subprocess.Popen(
            ["/bin/bash", str(driver_script)],
            cwd=str(root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        return False
    return True


def stop_driver(pids: list[int], dry_run: bool) -> None:
    if dry_run:
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            pass
    # short grace period
    deadline = time.time() + 4.0
    while time.time() < deadline:
        alive = []
        for pid in pids:
            if Path(f"/proc/{pid}").exists():
                alive.append(pid)
        if not alive:
            return
        time.sleep(0.2)
    for pid in pids:
        if Path(f"/proc/{pid}").exists():
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass


def emit_event(
    events_file: Path,
    log_file: Path,
    *,
    event: str,
    severity: str,
    payload: dict[str, Any],
    human: str,
) -> None:
    rec = {
        "ts": now_iso(),
        "event": event,
        "severity": severity,
        "payload": payload,
        "human": human,
    }
    append_jsonl(events_file, rec)
    append_log(log_file, f"{event} {human}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Progress guard agent for autonomous WFA loop.")
    p.add_argument("--root", default="", help="App root (`coint4/`).")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    root = Path(args.root).resolve() if str(args.root or "").strip() else Path(__file__).resolve().parents[2]
    aggregate_root = root / "artifacts" / "wfa" / "aggregate"
    state_dir = aggregate_root / ".autonomous"
    state_dir.mkdir(parents=True, exist_ok=True)

    driver_script = root / "scripts" / "optimization" / "autonomous_wfa_driver.sh"
    driver_log = state_dir / "driver.log"
    state_file = state_dir / "progress_guard_state.json"
    status_file = state_dir / "progress_guard_status.json"
    log_file = state_dir / "progress_guard.log"
    events_file = state_dir / "progress_guard_events.jsonl"

    cfg = {
        "driver_stale_sec": int(os.environ.get("PROGRESS_GUARD_DRIVER_STALE_SEC", "900")),
        "vps_down_alert_sec": int(os.environ.get("PROGRESS_GUARD_VPS_DOWN_ALERT_SEC", "600")),
        "vps_recover_after_sec": int(os.environ.get("PROGRESS_GUARD_VPS_RECOVER_AFTER_SEC", "900")),
        "vps_active_no_ssh_recover_sec": int(os.environ.get("PROGRESS_GUARD_VPS_ACTIVE_NO_SSH_RECOVER_SEC", "180")),
        "power_cycle_cooldown_sec": int(os.environ.get("PROGRESS_GUARD_POWER_CYCLE_COOLDOWN_SEC", "1800")),
        "power_cycle_max_per_hour": int(os.environ.get("PROGRESS_GUARD_POWER_CYCLE_MAX_PER_HOUR", "2")),
        "power_cycle_shutdown_wait_sec": int(os.environ.get("PROGRESS_GUARD_POWER_CYCLE_SHUTDOWN_WAIT_SEC", "20")),
        "power_cycle_boot_wait_sec": int(os.environ.get("PROGRESS_GUARD_POWER_CYCLE_BOOT_WAIT_SEC", "600")),
        "restart_cooldown_sec": int(os.environ.get("PROGRESS_GUARD_RESTART_COOLDOWN_SEC", "600")),
        "restart_max_per_hour": int(os.environ.get("PROGRESS_GUARD_RESTART_MAX_PER_HOUR", "4")),
        "event_ttl_sec": int(os.environ.get("PROGRESS_GUARD_EVENT_TTL_SEC", "600")),
        "server_ip": os.environ.get("SERVER_IP", "85.198.90.128"),
        "server_user": os.environ.get("SERVER_USER", "root"),
    }

    now_i = now_epoch()
    state = load_json(
        state_file,
        {
            "state_version": 1,
            "last_restart_epoch": 0,
            "restart_epochs": [],
            "last_power_cycle_epoch": 0,
            "power_cycle_epochs": [],
            "vps_down_since_epoch": 0,
            "event_last_emit": {},
        },
    )
    if not isinstance(state, dict):
        state = {
            "state_version": 1,
            "last_restart_epoch": 0,
            "restart_epochs": [],
            "last_power_cycle_epoch": 0,
            "power_cycle_epochs": [],
            "vps_down_since_epoch": 0,
            "event_last_emit": {},
        }
    state.setdefault("restart_epochs", [])
    state.setdefault("power_cycle_epochs", [])
    state.setdefault("event_last_emit", {})

    pending = count_pending(aggregate_root)
    driver_pids = find_driver_pids(driver_script)
    driver_running = len(driver_pids) > 0
    local_runner_count = count_local_queue_runners()
    last_log_epoch = last_driver_log_epoch(driver_log)
    since_last_log = (now_i - last_log_epoch) if last_log_epoch > 0 else None

    vps_ok = ssh_reachable(cfg["server_user"], cfg["server_ip"])
    serverspace_meta: dict[str, str] = {}
    serverspace_state = ""
    serverspace_server_id = ""
    down_since = int(float(state.get("vps_down_since_epoch", 0) or 0))
    power_cycled = False
    power_cycle_reason = ""
    if vps_ok:
        if down_since > 0 and should_emit(state, "vps_recovered", now_i, cfg["event_ttl_sec"]):
            emit_event(
                events_file,
                log_file,
                event="PROGRESS_GUARD_VPS_RECOVERED",
                severity="info",
                payload={"down_for_sec": max(0, now_i - down_since)},
                human="VPS снова доступен по SSH",
            )
        state["vps_down_since_epoch"] = 0
    else:
        if down_since <= 0:
            down_since = now_i
            state["vps_down_since_epoch"] = down_since
        down_for = now_i - down_since
        serverspace_meta = _serverspace_find(root, cfg["server_ip"])
        serverspace_state = str(serverspace_meta.get("state") or "").strip()
        serverspace_server_id = str(serverspace_meta.get("server_id") or "").strip()
        if pending > 0 and down_for >= cfg["vps_down_alert_sec"] and should_emit(state, "vps_unreachable", now_i, cfg["event_ttl_sec"]):
            emit_event(
                events_file,
                log_file,
                event="PROGRESS_GUARD_VPS_UNREACHABLE",
                severity="warning",
                payload={
                    "down_for_sec": down_for,
                    "pending": pending,
                    "serverspace_state": serverspace_state,
                    "server_id": serverspace_server_id,
                },
                human=f"VPS недоступен {down_for}s при pending={pending} (state={serverspace_state or 'unknown'})",
            )
        active_no_ssh_recover = (
            pending > 0
            and local_runner_count == 0
            and down_for >= cfg["vps_active_no_ssh_recover_sec"]
            and serverspace_state.strip().lower() == "active"
        )
        should_recover = (
            pending > 0
            and local_runner_count == 0
            and down_for >= cfg["vps_recover_after_sec"]
        )
        if active_no_ssh_recover or should_recover:
            ok, why = can_power_cycle(
                state,
                now_i,
                cfg["power_cycle_cooldown_sec"],
                cfg["power_cycle_max_per_hour"],
            )
            if ok:
                cycle_ok, cycle_reason = power_cycle_vps(
                    root,
                    server_user=cfg["server_user"],
                    server_ip=cfg["server_ip"],
                    shutdown_wait_sec=cfg["power_cycle_shutdown_wait_sec"],
                    boot_wait_sec=cfg["power_cycle_boot_wait_sec"],
                    dry_run=args.dry_run,
                )
                power_cycled = cycle_ok
                power_cycle_reason = cycle_reason
                state["last_power_cycle_epoch"] = now_i
                state.setdefault("power_cycle_epochs", []).append(now_i)
                if cycle_ok:
                    state["vps_down_since_epoch"] = 0
                    vps_ok = True
                    emit_event(
                        events_file,
                        log_file,
                        event="PROGRESS_GUARD_VPS_POWER_CYCLE_RECOVERED",
                        severity="warning",
                        payload={
                            "pending": pending,
                            "down_for_sec": down_for,
                            "reason": cycle_reason,
                            "serverspace_state": serverspace_state,
                            "server_id": serverspace_server_id,
                            "recovery_mode": "active_no_ssh" if active_no_ssh_recover else "default",
                        },
                        human=(
                            "выполнен power-cycle VPS, SSH восстановлен"
                            + (" (active+no-ssh fast path)" if active_no_ssh_recover else "")
                        ),
                    )
                else:
                    emit_event(
                        events_file,
                        log_file,
                        event="PROGRESS_GUARD_VPS_POWER_CYCLE_FAILED",
                        severity="error",
                        payload={
                            "pending": pending,
                            "down_for_sec": down_for,
                            "reason": cycle_reason,
                            "serverspace_state": serverspace_state,
                            "server_id": serverspace_server_id,
                            "recovery_mode": "active_no_ssh" if active_no_ssh_recover else "default",
                        },
                        human=f"power-cycle VPS не восстановил SSH ({cycle_reason})",
                    )
            elif should_emit(state, "power_cycle_throttled", now_i, cfg["event_ttl_sec"]):
                emit_event(
                    events_file,
                    log_file,
                    event="PROGRESS_GUARD_POWER_CYCLE_THROTTLED",
                    severity="warning",
                    payload={
                        "pending": pending,
                        "down_for_sec": down_for,
                        "reason": why,
                        "serverspace_state": serverspace_state,
                        "server_id": serverspace_server_id,
                        "recovery_mode": "active_no_ssh" if active_no_ssh_recover else "default",
                    },
                    human=(
                        f"power-cycle отложен ({why})"
                        + (" [active+no-ssh fast path]" if active_no_ssh_recover else "")
                    ),
                )

    restarted = False
    restart_reason = ""

    # Start missing driver if work exists.
    if pending > 0 and not driver_running:
        ok, why = can_restart(state, now_i, cfg["restart_cooldown_sec"], cfg["restart_max_per_hour"])
        if ok:
            if start_driver(root, driver_script, args.dry_run):
                state["last_restart_epoch"] = now_i
                state["restart_epochs"].append(now_i)
                restarted = True
                restart_reason = "driver_missing_pending"
                emit_event(
                    events_file,
                    log_file,
                    event="PROGRESS_GUARD_DRIVER_STARTED",
                    severity="warning",
                    payload={"pending": pending, "reason": restart_reason},
                    human="драйвер не работал при pending>0, выполнен старт",
                )
        elif should_emit(state, "restart_throttled_missing", now_i, cfg["event_ttl_sec"]):
            emit_event(
                events_file,
                log_file,
                event="PROGRESS_GUARD_RESTART_THROTTLED",
                severity="warning",
                payload={"reason": why, "pending": pending},
                human=f"рестарт драйвера отложен ({why})",
            )

    # Restart stale driver if there is work and no local queue runner progress.
    if pending > 0 and driver_running and since_last_log is not None:
        if since_last_log >= cfg["driver_stale_sec"] and local_runner_count == 0:
            ok, why = can_restart(state, now_i, cfg["restart_cooldown_sec"], cfg["restart_max_per_hour"])
            if ok:
                stop_driver(driver_pids, args.dry_run)
                if start_driver(root, driver_script, args.dry_run):
                    state["last_restart_epoch"] = now_i
                    state["restart_epochs"].append(now_i)
                    restarted = True
                    restart_reason = "driver_stale_no_local_runner"
                    emit_event(
                        events_file,
                        log_file,
                        event="PROGRESS_GUARD_DRIVER_RESTARTED",
                        severity="warning",
                        payload={"pending": pending, "since_last_log_sec": since_last_log, "reason": restart_reason},
                        human=f"драйвер stale ({since_last_log}s), выполнен рестарт",
                    )
            elif should_emit(state, "restart_throttled_stale", now_i, cfg["event_ttl_sec"]):
                emit_event(
                    events_file,
                    log_file,
                    event="PROGRESS_GUARD_RESTART_THROTTLED",
                    severity="warning",
                    payload={"reason": why, "pending": pending, "since_last_log_sec": since_last_log},
                    human=f"рестарт stale-драйвера отложен ({why})",
                )

    prune_restart_epochs(state, now_i)
    items = state.get("power_cycle_epochs")
    if not isinstance(items, list):
        items = []
    state["power_cycle_epochs"] = [int(x) for x in items if isinstance(x, (int, float)) and now_i - int(x) <= 3600]

    status = {
        "ts": now_iso(),
        "pending": pending,
        "driver_running": driver_running,
        "driver_pids": driver_pids,
        "local_runner_count": local_runner_count,
        "last_driver_log_epoch": last_log_epoch,
        "since_last_driver_log_sec": since_last_log,
        "vps_reachable": vps_ok,
        "serverspace_state": serverspace_state,
        "serverspace_server_id": serverspace_server_id,
        "vps_down_since_epoch": int(float(state.get("vps_down_since_epoch", 0) or 0)),
        "power_cycled": power_cycled,
        "power_cycle_reason": power_cycle_reason,
        "power_cycle_count_1h": len(state.get("power_cycle_epochs", [])),
        "restarted": restarted,
        "restart_reason": restart_reason,
        "restart_count_1h": len(state.get("restart_epochs", [])),
        "cfg": cfg,
    }

    dump_json(status_file, status)
    if not args.dry_run:
        dump_json(state_file, state)
    append_log(
        log_file,
        (
            "check "
            f"pending={pending} driver_running={driver_running} local_runner_count={local_runner_count} "
            f"vps_reachable={vps_ok} since_last_log_sec={since_last_log} "
            f"power_cycled={power_cycled} restarted={restarted}"
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
