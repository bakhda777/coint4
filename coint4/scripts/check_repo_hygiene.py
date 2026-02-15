#!/usr/bin/env python3
"""
Repo hygiene check: prevent accidentally tracking large/generated artifacts.

This script inspects the current HEAD tree (tracked files only) and fails if:
- any tracked path is in forbidden locations (outputs/, coint4/outputs/, coint4/data_downloaded/ except .gitkeep, etc.)
- any tracked file is above a size threshold (default: 3 MiB)
- any tracked file matches clearly-generated extensions (zip/db/sqlite/parquet/log/pid)

Rationale: SYNC_UP=1 syncs tracked files only. If a heavy artifact is tracked, it will be rsynced to the VPS.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class TrackedFile:
    path: str
    size_bytes: int


def _run(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, text=True)


def _git_root() -> str:
    return _run(["git", "rev-parse", "--show-toplevel"]).strip()


def _iter_tracked_files_index() -> Iterable[TrackedFile]:
    """
    Iterate tracked files from the Git index (staged state).

    This is the most actionable view for "clean cycle" work because SYNC_UP=1 syncs tracked files
    and we want the check to pass before committing.
    """
    raw = subprocess.check_output(["git", "ls-files", "-s", "-z"])
    if not raw:
        return

    # Entries: "<mode> <sha> <stage>\t<path>\0"
    entries = [e for e in raw.split(b"\0") if e]
    shas: List[str] = []
    paths: List[str] = []
    for ent in entries:
        try:
            meta, path_b = ent.split(b"\t", 1)
            parts = meta.split()
            if len(parts) < 3:
                continue
            sha = parts[1].decode("ascii", errors="ignore")
            path = path_b.decode("utf-8", errors="surrogateescape")
            if not sha or not path:
                continue
            shas.append(sha)
            paths.append(path)
        except Exception:
            continue

    if not shas:
        return

    proc = subprocess.Popen(
        ["git", "cat-file", "--batch-check=%(objectsize)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    for sha in shas:
        proc.stdin.write(sha + "\n")
    proc.stdin.close()

    sizes: List[int] = []
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            sizes.append(int(line))
        except ValueError:
            sizes.append(0)

    proc.wait()

    # Best-effort: zip by order; if mismatch, stop at min length.
    for path, size in zip(paths, sizes):
        yield TrackedFile(path=path, size_bytes=size)


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check repo for accidentally tracked heavy/generated files.")
    parser.add_argument("--max-mib", type=float, default=3.0, help="Fail if any tracked file exceeds this size (MiB).")
    args = parser.parse_args()

    # Canonical forbidden locations. (We check tracked HEAD state, not the working tree.)
    forbidden_prefixes = (
        "outputs/",
        "coint4/outputs/",
        ".ralph-tui/iterations/",
        ".secrets/",
    )

    forbidden_globs = (
        "*.parquet",
        "*.parquet.*",  # e.g. .parquet.tmp, .parquet.something
        "*.db",
        "*.db-shm",
        "*.db-wal",
        "*.sqlite",
        "*.zip",
        "*.tar",
        "*.tar.gz",
        "*.tgz",
        "*.log",
        "*.pid",
    )

    max_bytes = int(args.max_mib * 1024 * 1024)
    failures: List[str] = []

    for tf in _iter_tracked_files_index():
        p = tf.path

        if p.startswith(forbidden_prefixes):
            failures.append(f"forbidden path tracked: {p}")
            continue

        if p.startswith("coint4/data_downloaded/") and p != "coint4/data_downloaded/.gitkeep":
            failures.append(f"downloaded data must not be tracked: {p}")
            continue

        if _matches_any(p, forbidden_globs):
            failures.append(f"generated/large extension tracked: {p}")
            continue

        if tf.size_bytes > max_bytes:
            mib = tf.size_bytes / (1024 * 1024)
            failures.append(f"tracked file too large ({mib:.2f} MiB > {args.max_mib:.2f} MiB): {p}")

    if failures:
        root = os.path.relpath(_git_root(), os.getcwd())
        print(f"[hygiene] FAIL ({len(failures)}): repo root={root}")
        for msg in failures:
            print(f"- {msg}")
        print("")
        print("Fix suggestions:")
        print("- remove from index: git rm --cached <path>   (or git rm <path> if you want it deleted)")
        print("- ensure ignore rules cover the path/pattern")
        return 2

    print("[hygiene] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
