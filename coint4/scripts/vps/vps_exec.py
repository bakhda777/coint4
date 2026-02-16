#!/usr/bin/env python3
"""SSH exec helper for the VPS (optionally via a single tmux session)."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


DEFAULT_HOST = "85.198.90.128"
DEFAULT_USER = "root"
DEFAULT_KEY = str(Path.home() / ".ssh" / "id_ed25519")


@dataclass(frozen=True)
class SSHTarget:
    user: str
    host: str
    key_path: str


def _ssh_base_args(t: SSHTarget) -> List[str]:
    return [
        "ssh",
        "-i",
        t.key_path,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=5",
        f"{t.user}@{t.host}",
    ]


def ssh_exec(t: SSHTarget, cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    remote = f"bash -lc {shlex.quote(cmd)}"
    return subprocess.run(
        _ssh_base_args(t) + [remote],
        text=True,
        check=check,
    )


def tmux_has_session(t: SSHTarget, session: str) -> bool:
    cp = ssh_exec(t, f"tmux has-session -t {shlex.quote(session)}", check=False)
    return cp.returncode == 0


def tmux_exec(t: SSHTarget, session: str, cmd: str, *, create: bool = True) -> None:
    q_session = shlex.quote(session)
    q_cmd = shlex.quote(f"bash -lc {shlex.quote(cmd)}")

    if create:
        remote = (
            f"tmux has-session -t {q_session} 2>/dev/null || "
            f"tmux new-session -d -s {q_session}; "
            f"tmux send-keys -t {q_session} {q_cmd} C-m"
        )
    else:
        remote = f"tmux send-keys -t {q_session} {q_cmd} C-m"
    ssh_exec(t, remote, check=True)


def _join_command(parts: List[str]) -> str:
    if parts and parts[0] == "--":
        parts = parts[1:]
    return " ".join(parts).strip()


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--key", default=DEFAULT_KEY)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exec = sub.add_parser("exec", help="Run a single command over SSH (bash -lc).")
    p_exec.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")

    p_has = sub.add_parser("tmux-has", help="Check if a tmux session exists.")
    p_has.add_argument("--session", required=True)

    p_tmux = sub.add_parser("tmux-exec", help="Run a command inside a tmux session.")
    p_tmux.add_argument("--session", required=True)
    p_tmux.add_argument("--no-create", action="store_true", help="Do not create session if missing")
    p_tmux.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")

    args = parser.parse_args(argv)
    target = SSHTarget(user=args.user, host=args.host, key_path=args.key)

    if args.cmd == "exec":
        if not args.command:
            raise SystemExit("Missing command")
        cmd = _join_command(args.command)
        if not cmd:
            raise SystemExit("Missing command")
        cp = ssh_exec(target, cmd, check=False)
        if cp.stdout:
            print(cp.stdout, end="")
        if cp.stderr:
            print(cp.stderr, end="")
        return cp.returncode

    if args.cmd == "tmux-has":
        ok = tmux_has_session(target, args.session)
        return 0 if ok else 2

    if args.cmd == "tmux-exec":
        if not args.command:
            raise SystemExit("Missing command")
        cmd = _join_command(args.command)
        if not cmd:
            raise SystemExit("Missing command")
        tmux_exec(target, args.session, cmd, create=not args.no_create)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
