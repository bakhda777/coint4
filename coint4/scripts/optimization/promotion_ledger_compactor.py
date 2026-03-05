#!/usr/bin/env python3
"""Compact append-only promotion ledger while preserving history in gzip archives."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import os
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_int(raw: str | None, default: int) -> int:
    try:
        return int(float(raw or default))
    except Exception:
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description="Compact promotion ledger jsonl.")
    parser.add_argument("--root", default="", help="App root (`coint4/`).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve() if str(args.root or "").strip() else Path(__file__).resolve().parents[2]
    state_dir = root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    ledger_path = state_dir / "promotion_ledger.jsonl"
    archive_dir = state_dir / "archive"
    lock_path = state_dir / "promotion_ledger_compactor.lock"
    log_path = state_dir / "promotion_ledger_compactor.log"

    keep_lines = parse_int(os.environ.get("PROMOTION_LEDGER_KEEP_LINES"), 5000)
    min_lines = parse_int(os.environ.get("PROMOTION_LEDGER_COMPACT_MIN_LINES"), 10000)

    state_dir.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return 0

        if not ledger_path.exists():
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{utc_now_iso()} | skip missing_ledger\n")
            return 0

        try:
            lines = ledger_path.read_text(encoding="utf-8").splitlines(keepends=True)
        except Exception as exc:  # noqa: BLE001
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{utc_now_iso()} | error read_failed={type(exc).__name__}:{exc}\n")
            return 1

        total = len(lines)
        if total < min_lines:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{utc_now_iso()} | skip total_lines={total} min_lines={min_lines}\n")
            return 0

        keep = max(1, keep_lines)
        archived_lines = lines[:-keep]
        kept_lines = lines[-keep:]

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"promotion_ledger_{ts}.jsonl.gz"

        if not args.dry_run:
            with gzip.open(archive_path, "wt", encoding="utf-8") as handle:
                handle.writelines(archived_lines)
            ledger_path.write_text("".join(kept_lines), encoding="utf-8")

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{utc_now_iso()} | compacted total={total} archived={len(archived_lines)} kept={len(kept_lines)} "
                f"archive={archive_path.name} dry_run={int(bool(args.dry_run))}\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
