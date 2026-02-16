#!/usr/bin/env python3
"""Minimal Serverspace Public API client (no secrets in logs).

Auth:
  - env: SERVSPACE_API_KEY
  - or file: <repo_root>/.secrets/serverspace_api_key (chmod 600, gitignored)

Default API base:
  https://api.serverspace.ru/api/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_API_BASE = "https://api.serverspace.ru/api/v1"
DEFAULT_TARGET_IP = "85.198.90.128"


def _repo_root() -> Path:
    # .../coint4/scripts/vps/serverspace_api.py -> app_root is parents[2], repo_root is parent
    return Path(__file__).resolve().parents[3]


def _load_api_key() -> str:
    key = os.environ.get("SERVSPACE_API_KEY", "").strip()
    if key:
        return key
    key_path = _repo_root() / ".secrets" / "serverspace_api_key"
    if key_path.exists():
        file_key = key_path.read_text(encoding="utf-8").strip()
        if file_key:
            return file_key
        raise RuntimeError(".secrets/serverspace_api_key is empty")
    raise RuntimeError("SERVSPACE_API_KEY not set and .secrets/serverspace_api_key not found")


def _request_json(
    *,
    method: str,
    url: str,
    api_key: str,
    body: Optional[Dict[str, Any]] = None,
    timeout_sec: int = 30,
) -> Any:
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, method=method.upper())
    req.add_header("Content-Type", "application/json")
    req.add_header("X-API-KEY", api_key)
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} for {method} {url}: {raw[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Request failed for {method} {url}: {e}") from e

    try:
        return json.loads(raw) if raw else None
    except json.JSONDecodeError:
        return raw


def _extract_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("data", "items", "servers"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def list_servers(*, api_base: str, api_key: str) -> List[Dict[str, Any]]:
    payload = _request_json(method="GET", url=f"{api_base}/servers", api_key=api_key)
    return _extract_list(payload)


def _iter_possible_ips(server: Dict[str, Any]) -> Iterable[str]:
    for key in (
        "ip",
        "ipv4",
        "ip_address",
        "address",
        "public_ip",
        "publicIp",
        "public_ip_address",
        "publicIpAddress",
    ):
        value = server.get(key)
        if isinstance(value, str):
            yield value
    # Some APIs embed addresses as lists/dicts.
    for key in ("addresses", "networks", "ips", "ipAddresses", "nics", "interfaces"):
        value = server.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    yield item
                if isinstance(item, dict):
                    for v in item.values():
                        if isinstance(v, str):
                            yield v
        if isinstance(value, dict):
            for v in value.values():
                if isinstance(v, str):
                    yield v


@dataclass(frozen=True)
class ServerMatch:
    server_id: str
    name: str
    state: str


def find_server_by_ip(*, api_base: str, api_key: str, ip: str) -> Optional[ServerMatch]:
    for srv in list_servers(api_base=api_base, api_key=api_key):
        if ip not in set(_iter_possible_ips(srv)):
            continue
        server_id = str(srv.get("id") or srv.get("uuid") or srv.get("server_id") or srv.get("serverId") or "")
        name = str(srv.get("name") or srv.get("hostname") or "")
        state = str(srv.get("state") or srv.get("status") or "")
        if server_id:
            return ServerMatch(server_id=server_id, name=name, state=state)
    return None


def power_on(*, api_base: str, api_key: str, server_id: str) -> None:
    try:
        _request_json(
            method="POST",
            url=f"{api_base}/servers/{server_id}/power/on",
            api_key=api_key,
            body={"server_id": server_id},
        )
    except RuntimeError as e:
        # Serverspace returns HTTP 400 with message "The server is already on".
        if "already on" in str(e).lower():
            return
        raise


def shutdown(*, api_base: str, api_key: str, server_id: str) -> None:
    try:
        _request_json(
            method="POST",
            url=f"{api_base}/servers/{server_id}/power/shutdown",
            api_key=api_key,
            body={"server_id": server_id},
        )
    except RuntimeError as e:
        # Serverspace may return an error if the server is already stopped.
        text = str(e).lower()
        if "already" in text and ("off" in text or "stopped" in text or "shutdown" in text):
            return
        raise


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--ip", default=DEFAULT_TARGET_IP, help="Target VPS public IP for find")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List servers")
    sub.add_parser("find", help="Find server id/name/state by --ip")
    p_on = sub.add_parser("power-on", help="Power on server by id")
    p_on.add_argument("--id", required=True)
    p_off = sub.add_parser("shutdown", help="Shutdown server by id")
    p_off.add_argument("--id", required=True)

    args = parser.parse_args(argv)
    api_key = _load_api_key()

    if args.cmd == "list":
        servers = list_servers(api_base=args.api_base, api_key=api_key)
        # Print minimal fields only.
        for srv in servers:
            sid = srv.get("id") or srv.get("uuid") or srv.get("server_id") or srv.get("serverId") or ""
            name = srv.get("name") or srv.get("hostname") or ""
            state = srv.get("state") or srv.get("status") or ""
            print(json.dumps({"id": sid, "name": name, "state": state}, ensure_ascii=False))
        return 0

    if args.cmd == "find":
        match = find_server_by_ip(api_base=args.api_base, api_key=api_key, ip=args.ip)
        if match is None:
            print(json.dumps({"ip": args.ip, "found": False}, ensure_ascii=False))
            return 2
        print(json.dumps({"ip": args.ip, "found": True, **match.__dict__}, ensure_ascii=False))
        return 0

    if args.cmd == "power-on":
        power_on(api_base=args.api_base, api_key=api_key, server_id=args.id)
        print(json.dumps({"id": args.id, "action": "power-on", "ok": True}))
        return 0

    if args.cmd == "shutdown":
        shutdown(api_base=args.api_base, api_key=api_key, server_id=args.id)
        print(json.dumps({"id": args.id, "action": "shutdown", "ok": True}))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
