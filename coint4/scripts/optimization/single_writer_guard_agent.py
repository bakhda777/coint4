#!/usr/bin/env python3
"""Ensure single-writer orchestration processes and critical timers are active."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value or 0))
    except Exception:
        return default


def proc_start_ticks(pid: int) -> int:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="ignore")
        # field 22 in /proc/<pid>/stat (1-based)
        return int(text.split()[21])
    except Exception:
        return 0


def find_matching_pids(pattern: str) -> list[int]:
    out: list[int] = []
    self_pid = os.getpid()
    parent_pid = os.getppid()
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid in {self_pid, parent_pid}:
            continue
        try:
            cmd = (
                Path(f"/proc/{pid}/cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode("utf-8", "ignore")
                .strip()
            )
        except Exception:
            continue
        if not cmd:
            continue
        if "python3 - <<" in cmd or "pgrep -f" in cmd:
            continue
        if pattern in cmd:
            out.append(pid)
    return out


def terminate_extra(pids: list[int]) -> tuple[list[int], list[int]]:
    if len(pids) <= 1:
        return pids, []
    ordered = sorted(pids, key=proc_start_ticks)
    keep = ordered[0]
    killed: list[int] = []
    for pid in ordered[1:]:
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except Exception:
            continue
    return [keep], killed


def ensure_unit_active(unit: str) -> tuple[bool, str]:
    if not shutil_which("systemctl"):
        return False, "systemctl_missing"
    chk = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True, check=False)
    active = chk.returncode == 0 and (chk.stdout or "").strip() == "active"
    if active:
        return True, "already_active"

    enable = subprocess.run(
        ["systemctl", "--user", "enable", "--now", unit],
        capture_output=True,
        text=True,
        check=False,
    )
    if enable.returncode != 0:
        return False, f"enable_failed:{(enable.stderr or '').strip()[-200:]}"

    chk2 = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True, check=False)
    active2 = chk2.returncode == 0 and (chk2.stdout or "").strip() == "active"
    return active2, "enabled" if active2 else "post_check_failed"


def shutil_which(cmd: str) -> str | None:
    for path in os.environ.get("PATH", "").split(":"):
        candidate = Path(path) / cmd
        try:
            if candidate.exists() and os.access(candidate, os.X_OK):
                return str(candidate)
        except Exception:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Single writer guard for autonomous orchestration.")
    parser.add_argument("--root", default="", help="App root (`coint4/`).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve() if str(args.root or "").strip() else Path(__file__).resolve().parents[2]
    state_dir = root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    lock_path = state_dir / "single_writer_guard.lock"
    log_path = state_dir / "single_writer_guard.log"

    state_dir.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return 0

        patterns = {
            "driver": "scripts/optimization/autonomous_wfa_driver.sh",
            "confirm_dispatch": "scripts/optimization/confirm_dispatch_agent.sh",
            "gatekeeper": "scripts/optimization/promotion_gatekeeper_agent.py",
        }

        killed_total = 0
        kept_total = 0
        lines: list[str] = []
        for name, pattern in patterns.items():
            pids = find_matching_pids(pattern)
            if len(pids) <= 1:
                kept_total += len(pids)
                lines.append(f"proc {name} pids={pids}")
                continue
            keep, killed = terminate_extra(pids) if not args.dry_run else ([min(pids)], sorted(pids)[1:])
            kept_total += len(keep)
            killed_total += len(killed)
            lines.append(f"proc {name} keep={keep} killed={killed}")

        units = [
            "autonomous-wfa-driver.service",
            "autonomous-wfa-watchdog.timer",
            "autonomous-queue-seeder.timer",
            "autonomous-confirm-dispatch-agent.timer",
            "autonomous-search-director-agent.timer",
            "autonomous-promotion-gatekeeper-agent.timer",
        ]
        if os.environ.get("SINGLE_WRITER_GUARD_ENSURE_NEW_UNITS", "1").strip().lower() in {"1", "true", "yes", "on"}:
            units.extend(
                [
                    "autonomous-contract-auditor-agent.timer",
                    "autonomous-vps-capacity-controller-agent.timer",
                    "autonomous-confirm-diversity-guard-agent.timer",
                    "autonomous-deterministic-error-blacklist-agent.timer",
                    "autonomous-confirm-sla-escalator-agent.timer",
                    "autonomous-single-writer-guard-agent.timer",
                    "autonomous-promotion-ledger-compactor.timer",
                ]
            )

        inactive = 0
        for unit in units:
            ok, reason = ensure_unit_active(unit) if not args.dry_run else (True, "dry_run")
            if not ok:
                inactive += 1
            lines.append(f"unit {unit} ok={int(ok)} reason={reason}")

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{utc_now_iso()} | kept={kept_total} killed={killed_total} inactive_units={inactive} dry_run={int(bool(args.dry_run))}\n"
            )
            for line in lines:
                handle.write(f"{utc_now_iso()} | {line}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
