#!/usr/bin/env python3
"""Run quick verification commands on the VPS (no heavy backtests)."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_HOST = "85.198.90.128"
DEFAULT_USER = "root"
DEFAULT_KEY = str(Path.home() / ".ssh" / "id_ed25519")
DEFAULT_REMOTE_REPO_DIR = "/opt/coint4"


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


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--remote-repo-dir", default=DEFAULT_REMOTE_REPO_DIR)
    args = parser.parse_args(argv)

    remote_cmd = f"cd {args.remote_repo_dir} && make ci"
    remote = f"bash -lc {shlex.quote(remote_cmd)}"
    cp = subprocess.run(
        _ssh_base(args.user, args.host, args.key) + [remote],
        check=False,
        text=True,
    )
    if cp.returncode != 0:
        print(
            "[vps_verify] ERROR: verify stage failed. "
            "If dependencies are missing, run on VPS: `cd /opt/coint4 && make setup`.",
            file=sys.stderr,
        )
    return cp.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
