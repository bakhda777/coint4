#!/usr/bin/env python3
"""Run loop with explicit VPS API power cycle.

Sequence:
1) Power on VPS via Serverspace API.
2) Wait until SSH is ready.
3) Run `make preflight-loop`.
4) Run `make vps-baseline` with SKIP_POWER=1 and STOP_AFTER=0.
5) Power off VPS via Serverspace API (always, even on failure).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_HOST = "85.198.90.128"
DEFAULT_USER = "root"
DEFAULT_SSH_KEY = str(Path.home() / ".ssh" / "id_ed25519")
DEFAULT_TIMEOUT_SEC = 900
DEFAULT_POLL_SEC = 5


def _repo_root() -> Path:
    # .../coint4/scripts/optimization/run_loop_with_api_power.py -> repo root at parents[3]
    return Path(__file__).resolve().parents[3]


def _load_serverspace_module() -> Any:
    module_path = _repo_root() / "coint4" / "scripts" / "vps" / "serverspace_api.py"
    spec = importlib.util.spec_from_file_location("serverspace_api", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Serverspace module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _server_id_from_payload(server: Dict[str, Any]) -> str:
    for key in ("id", "uuid", "server_id", "serverId"):
        value = server.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _resolve_server_id(
    *,
    serverspace: Any,
    api_base: str,
    api_key: str,
    host: str,
    explicit_id: str,
    explicit_name: str,
) -> str:
    server_id = explicit_id.strip()
    if server_id:
        return server_id

    servers = serverspace.list_servers(api_base=api_base, api_key=api_key)

    target_name = explicit_name.strip()
    if target_name:
        for server in servers:
            name = str(server.get("name") or "")
            hostname = str(server.get("hostname") or "")
            if target_name in (name, hostname):
                sid = _server_id_from_payload(server)
                if sid:
                    return sid

    match = serverspace.find_server_by_ip(api_base=api_base, api_key=api_key, ip=host)
    if match is not None and getattr(match, "server_id", ""):
        return str(match.server_id).strip()

    if len(servers) == 1:
        sid = _server_id_from_payload(servers[0])
        if sid:
            return sid

    raise RuntimeError(
        f"Could not resolve Serverspace server id (host={host}). "
        "Set SERVER_ID or SERVER_NAME."
    )


def _ssh_probe(*, user: str, host: str, ssh_key: str) -> bool:
    cmd = [
        "ssh",
        "-i",
        ssh_key,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=5",
        f"{user}@{host}",
        "echo ok",
    ]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _wait_for_ssh(*, user: str, host: str, ssh_key: str, timeout_sec: int, poll_sec: int) -> None:
    key_path = Path(ssh_key).expanduser()
    if not key_path.exists():
        raise RuntimeError(f"SSH key not found: {key_path}")

    start = time.monotonic()
    while True:
        if _ssh_probe(user=user, host=host, ssh_key=str(key_path)):
            return
        if time.monotonic() - start >= timeout_sec:
            raise RuntimeError(
                f"SSH is not ready for {user}@{host} after {timeout_sec} seconds"
            )
        time.sleep(max(1, poll_sec))


def _run_make(target: str, *, env: Dict[str, str], cwd: Path) -> None:
    cmd = ["make", target]
    completed = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` failed with exit code {completed.returncode}")


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("SERVER_IP", DEFAULT_HOST))
    parser.add_argument("--user", default=os.environ.get("SERVER_USER", DEFAULT_USER))
    parser.add_argument("--ssh-key", default=os.environ.get("SSH_KEY", DEFAULT_SSH_KEY))
    parser.add_argument("--server-id", default=os.environ.get("SERVER_ID", ""))
    parser.add_argument("--server-name", default=os.environ.get("SERVER_NAME", ""))
    parser.add_argument("--api-base", default=os.environ.get("SERVSPACE_API_BASE", ""))
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--poll-sec", type=int, default=DEFAULT_POLL_SEC)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    repo_root = _repo_root()
    serverspace = _load_serverspace_module()
    api_base = args.api_base or serverspace.DEFAULT_API_BASE

    try:
        api_key = serverspace._load_api_key()  # noqa: SLF001 - local helper from sibling script
    except Exception as exc:
        print(f"ERROR: failed to load SERVSPACE_API_KEY: {exc}", file=sys.stderr)
        return 1

    try:
        server_id = _resolve_server_id(
            serverspace=serverspace,
            api_base=api_base,
            api_key=api_key,
            host=args.host,
            explicit_id=args.server_id,
            explicit_name=args.server_name,
        )
    except Exception as exc:
        print(f"ERROR: failed to resolve server id: {exc}", file=sys.stderr)
        return 1

    exit_code = 0
    powered_on = False

    try:
        print(f"[loop] API power on: server_id={server_id} host={args.host}")
        serverspace.power_on(api_base=api_base, api_key=api_key, server_id=server_id)
        powered_on = True

        print(f"[loop] waiting for SSH {args.user}@{args.host}")
        _wait_for_ssh(
            user=args.user,
            host=args.host,
            ssh_key=args.ssh_key,
            timeout_sec=max(1, args.timeout_sec),
            poll_sec=max(1, args.poll_sec),
        )
        print("[loop] SSH is ready")

        if not args.skip_preflight:
            print("[loop] running: make preflight-loop")
            _run_make("preflight-loop", env=os.environ.copy(), cwd=repo_root)

        if not args.skip_baseline:
            print("[loop] running: make vps-baseline (SKIP_POWER=1 STOP_AFTER=0)")
            env = os.environ.copy()
            env.setdefault("SKIP_POWER", "1")
            env.setdefault("STOP_AFTER", "0")
            _run_make("vps-baseline", env=env, cwd=repo_root)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        if powered_on:
            print(f"[loop] API power off: server_id={server_id}")
            try:
                serverspace.shutdown(api_base=api_base, api_key=api_key, server_id=server_id)
            except Exception as exc:
                print(f"ERROR: failed to power off server {server_id}: {exc}", file=sys.stderr)
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
