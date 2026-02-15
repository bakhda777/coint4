#!/usr/bin/env python3
"""Start WFA queue runs on the VPS inside a single tmux session (no window explosion)."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_HOST = "85.198.90.128"
DEFAULT_USER = "root"
DEFAULT_KEY = str(Path.home() / ".ssh" / "id_ed25519")
DEFAULT_REMOTE_REPO_DIR = "/opt/coint4"
DEFAULT_TMUX_SESSION = "coint4-wfa"


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


def _tmux_run(user: str, host: str, key: str, session: str, cmd: str) -> None:
    q_session = shlex.quote(session)
    q_cmd = shlex.quote(f"bash -lc {shlex.quote(cmd)}")
    remote = (
        f"tmux has-session -t {q_session} 2>/dev/null && "
        f"tmux send-keys -t {q_session} {q_cmd} C-m || "
        f"tmux new-session -d -s {q_session} {q_cmd}"
    )
    subprocess.run(_ssh_base(user, host, key) + [remote], check=True, text=True)


def _load_manifest(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries") or []
    queues: List[str] = []
    seen = set()
    for e in entries:
        if not isinstance(e, dict):
            continue
        q = str(e.get("queue") or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        queues.append(q)
    return queues


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--remote-repo-dir", default=DEFAULT_REMOTE_REPO_DIR)
    parser.add_argument("--session", default=DEFAULT_TMUX_SESSION)
    parser.add_argument("--manifest", type=Path, required=True, help="Local JSON manifest from build_wfa_manifest.py")
    args = parser.parse_args(argv)

    queues = _load_manifest(args.manifest)
    if not queues:
        print("[vps_run_wfa] No queues found in manifest; nothing to run.")
        return 0

    # Build a single remote loop command so we keep one tmux session and one window.
    q_list = " ".join(shlex.quote(q) for q in queues)
    remote_cmd = f"""
set -euo pipefail
cd {shlex.quote(args.remote_repo_dir)}/coint4
echo "[vps_run_wfa] queues={len(queues)}"
for q in {q_list}; do
  echo "[vps_run_wfa] start queue=$q"
  bash scripts/optimization/watch_wfa_queue.sh --queue "$q"
done
mkdir -p outputs
date -u +"%Y-%m-%dT%H:%M:%SZ" > outputs/WFA_RUN_DONE.txt
echo "[vps_run_wfa] done"
""".strip()

    _tmux_run(args.user, args.host, args.key, args.session, remote_cmd)
    print(f"[vps_run_wfa] started in tmux session={args.session} host={args.user}@{args.host}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

