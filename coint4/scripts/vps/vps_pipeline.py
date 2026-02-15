#!/usr/bin/env python3
"""End-to-end local orchestrator: power -> wait-ssh -> sync -> verify -> run -> fetch.

This script runs on the *local* machine (this repo host) and uses:
  - Serverspace Public API (HTTPS) to power on/off the VPS
  - SSH/rsync to sync and operate the repo on the VPS
  - a single tmux session on the VPS for long-running WFA queues
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional


DEFAULT_HOST = "85.198.90.128"
DEFAULT_USER = "root"
DEFAULT_KEY = str(Path.home() / ".ssh" / "id_ed25519")
DEFAULT_REMOTE_REPO_DIR = "/opt/coint4"
DEFAULT_TMUX_SESSION = "coint4-wfa"


def _app_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _script_path(name: str) -> Path:
    return Path(__file__).resolve().with_name(name)


def _load_serverspace_module() -> Any:
    path = _script_path("serverspace_api.py")
    spec = importlib.util.spec_from_file_location("serverspace_api", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load serverspace_api.py")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses (and other libs) may look up the module by name during import time
    # via sys.modules[__module__]; register before exec_module.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ssh_probe(user: str, host: str, key: str) -> bool:
    cmd = [
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
        "echo ok",
    ]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def wait_for_ssh(*, user: str, host: str, key: str, timeout_sec: int = 900) -> None:
    start = time.time()
    while time.time() - start < timeout_sec:
        if _ssh_probe(user, host, key):
            return
        time.sleep(5)
    raise RuntimeError("SSH not ready after timeout")


def _run_py(script: str, args: List[str]) -> None:
    subprocess.run([sys.executable, str(_script_path(script))] + args, check=True)


def _remote_file_exists(*, user: str, host: str, key: str, path: str) -> bool:
    cmd = [
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
        f"test -f {path}",
    ]
    return (
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        == 0
    )


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Run power-on -> wait -> sync -> verify -> run -> fetch")
    parser.add_argument("--power-on", action="store_true")
    parser.add_argument("--wait-ssh", action="store_true")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--fetch", action="store_true")

    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--remote-repo-dir", default=DEFAULT_REMOTE_REPO_DIR)
    parser.add_argument("--tmux-session", default=DEFAULT_TMUX_SESSION)
    parser.add_argument("--server-id", default="", help="Serverspace server id (optional; auto-find by IP if empty)")
    parser.add_argument("--api-base", default="", help="Override Serverspace API base (optional)")
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--include-runs", action="store_true", help="Fetch heavy runs (off by default)")
    args = parser.parse_args(argv)

    if args.all:
        args.power_on = args.wait_ssh = args.sync = args.verify = args.run = True
        args.fetch = True

    serverspace = _load_serverspace_module()
    api_base = args.api_base or serverspace.DEFAULT_API_BASE

    if args.power_on:
        api_key = serverspace._load_api_key()  # noqa: SLF001 (local-only helper)
        server_id = args.server_id
        if not server_id:
            match = serverspace.find_server_by_ip(api_base=api_base, api_key=api_key, ip=args.host)
            if match is None:
                raise RuntimeError(f"Could not find Serverspace server by IP: {args.host}")
            server_id = match.server_id
        serverspace.power_on(api_base=api_base, api_key=api_key, server_id=server_id)
        print(f"[vps_pipeline] power-on ok id={server_id}")

    if args.wait_ssh:
        print(f"[vps_pipeline] waiting for SSH {args.user}@{args.host} ...")
        wait_for_ssh(user=args.user, host=args.host, key=args.key, timeout_sec=args.timeout_sec)
        print("[vps_pipeline] ssh ready")

    if args.sync:
        _run_py(
            "sync_to_vps.py",
            [
                "--host",
                args.host,
                "--user",
                args.user,
                "--key",
                args.key,
                "--remote-repo-dir",
                args.remote_repo_dir,
            ],
        )

    if args.verify:
        _run_py(
            "vps_verify.py",
            [
                "--host",
                args.host,
                "--user",
                args.user,
                "--key",
                args.key,
                "--remote-repo-dir",
                args.remote_repo_dir,
            ],
        )

    manifest_path: Optional[Path] = None
    if args.run:
        app_root = _app_root()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        manifest_path = app_root / "outputs" / f"wfa_manifest_{stamp}.json"
        _run_py("build_wfa_manifest.py", ["--output", str(manifest_path)])
        _run_py(
            "vps_run_wfa.py",
            [
                "--host",
                args.host,
                "--user",
                args.user,
                "--key",
                args.key,
                "--remote-repo-dir",
                args.remote_repo_dir,
                "--session",
                args.tmux_session,
                "--manifest",
                str(manifest_path),
            ],
        )

    if args.fetch:
        sentinel = f"{args.remote_repo_dir.rstrip('/')}/coint4/outputs/WFA_RUN_DONE.txt"
        if not _remote_file_exists(user=args.user, host=args.host, key=args.key, path=sentinel):
            print("[vps_pipeline] sentinel not found; run likely still in progress.")
            print(f"[vps_pipeline] check: python3 coint4/scripts/vps/vps_exec.py tmux-has --session {args.tmux_session}")
            print(f"[vps_pipeline] attach: ssh -i {args.key} {args.user}@{args.host} 'tmux attach -t {args.tmux_session}'")
            print("[vps_pipeline] fetch later: python3 coint4/scripts/vps/vps_fetch_results.py")
            return 0
        _run_py(
            "vps_fetch_results.py",
            [
                "--host",
                args.host,
                "--user",
                args.user,
                "--key",
                args.key,
                "--remote-repo-dir",
                args.remote_repo_dir,
            ]
            + (["--include-runs"] if args.include_runs else []),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
