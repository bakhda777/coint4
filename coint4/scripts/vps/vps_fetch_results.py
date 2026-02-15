#!/usr/bin/env python3
"""Fetch lightweight results from the VPS into a gitignored local directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List


DEFAULT_HOST = "85.198.90.128"
DEFAULT_USER = "root"
DEFAULT_KEY = str(Path.home() / ".ssh" / "id_ed25519")
DEFAULT_REMOTE_REPO_DIR = "/opt/coint4"


def _app_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _rsync_base(user: str, host: str, key: str) -> List[str]:
    return [
        "rsync",
        "-az",
        "--human-readable",
        "--progress",
        "-e",
        f"ssh -i {key} -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5",
    ]


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--remote-repo-dir", default=DEFAULT_REMOTE_REPO_DIR)
    parser.add_argument("--include-runs", action="store_true", help="Also fetch artifacts/wfa/runs/** (huge)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local output dir (default: <app_root>/outputs/vps_fetch/<timestamp>/)",
    )
    args = parser.parse_args(argv)

    app_root = _app_root()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.output_dir or (app_root / "outputs" / "vps_fetch" / stamp)
    out.mkdir(parents=True, exist_ok=True)

    remote_app = f"{args.remote_repo_dir.rstrip('/')}/coint4"

    # Default: only light inputs/rollups/logs + outputs (sentinels, summaries, etc).
    pulls = [
        (f"{args.user}@{args.host}:{remote_app}/artifacts/wfa/aggregate/", out / "aggregate"),
        (f"{args.user}@{args.host}:{remote_app}/outputs/", out / "outputs"),
        (f"{args.user}@{args.host}:{args.remote_repo_dir.rstrip('/')}/SYNCED_FROM_COMMIT.txt", out / "SYNCED_FROM_COMMIT.txt"),
    ]

    for src, dst in pulls:
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(_rsync_base(args.user, args.host, args.key) + [src, str(dst)], check=True)

    if args.include_runs:
        dst = out / "runs"
        subprocess.run(
            _rsync_base(args.user, args.host, args.key)
            + [f"{args.user}@{args.host}:{remote_app}/artifacts/wfa/runs/", str(dst)],
            check=True,
        )

    print(f"[vps_fetch_results] ok output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
