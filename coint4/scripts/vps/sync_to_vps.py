#!/usr/bin/env python3
"""Sync code to VPS via rsync (excluding secrets/artifacts) and record git SHA."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_HOST = "85.198.90.128"
DEFAULT_USER = "root"
DEFAULT_KEY = str(Path.home() / ".ssh" / "id_ed25519")
DEFAULT_REMOTE_REPO_DIR = "/opt/coint4"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_sha(repo_root: Path) -> str:
    cp = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        check=True,
        capture_output=True,
    )
    return cp.stdout.strip()


def _ssh_base(user: str, host: str, key: str) -> List[str]:
    return [
        "ssh",
        "-i",
        key,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=5",
        f"{user}@{host}",
    ]


def _rsync_cmd(user: str, host: str, key: str, remote_dir: str, *, delete: bool, dry_run: bool) -> List[str]:
    cmd: List[str] = [
        "rsync",
        "-az",
        "--human-readable",
        "--progress",
        "-e",
        f"ssh -i {key} -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5",
    ]
    if delete:
        cmd.append("--delete")
    if dry_run:
        cmd.append("--dry-run")

    # Excludes: keep this conservative (never send secrets or huge generated dirs).
    excludes = [
        ".git/",
        ".secrets/",
        ".ralph-tui/",
        "coint4/.venv/",
        "coint4/.cache/",
        "coint4/.pytest_cache/",
        "coint4/.ruff_cache/",
        "coint4/.mypy_cache/",
        "coint4/__pycache__/",
        "coint4/outputs/",
        "coint4/results/",
        "coint4/bench/",
        "coint4/data_downloaded/",
        "coint4/data_optimized/",
        "**/*.log",
        "**/*.pid",
    ]
    for ex in excludes:
        cmd.extend(["--exclude", ex])

    # Allow small, tracked WFA queue/index inputs on the VPS, but never sync heavy run artifacts.
    includes = [
        "coint4/artifacts/",
        "coint4/artifacts/wfa/",
        "coint4/artifacts/wfa/aggregate/",
        "coint4/artifacts/wfa/aggregate/***",
    ]
    for inc in includes:
        cmd.extend(["--include", inc])
    cmd.extend(["--exclude", "coint4/artifacts/***"])

    repo_root = _repo_root()
    # Sync only the essential project surface.
    sources = [
        str(repo_root / "AGENTS.md"),
        str(repo_root / "Makefile"),
        str(repo_root / "docs"),
        str(repo_root / "coint4"),
    ]
    cmd.extend(sources)
    cmd.append(f"{user}@{host}:{remote_dir.rstrip('/')}/")
    return cmd


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--remote-repo-dir", default=DEFAULT_REMOTE_REPO_DIR)
    parser.add_argument("--delete", action="store_true", help="rsync --delete (careful)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    sha = _git_sha(repo_root)

    # Ensure remote dir exists.
    subprocess.run(
        _ssh_base(args.user, args.host, args.key)
        + [f"mkdir -p {args.remote_repo_dir} && echo ok"],
        check=True,
        text=True,
    )

    cmd = _rsync_cmd(
        args.user,
        args.host,
        args.key,
        args.remote_repo_dir,
        delete=args.delete,
        dry_run=args.dry_run,
    )
    subprocess.run(cmd, check=True)

    # Record local SHA on the VPS without printing secrets.
    subprocess.run(
        _ssh_base(args.user, args.host, args.key)
        + [f"printf '%s\\n' {sha} > {args.remote_repo_dir.rstrip('/')}/SYNCED_FROM_COMMIT.txt"],
        check=True,
        text=True,
    )

    print(f"[sync_to_vps] ok sha={sha} remote={args.user}@{args.host}:{args.remote_repo_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
