"""Serverspace power helpers used by powered WFA runner.

This module intentionally avoids third-party dependencies and does not log secrets.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


class ServerspaceError(RuntimeError):
    """Domain-specific error for Serverspace API operations."""


def _repo_root() -> Path:
    # .../coint4/src/coint2/ops/serverspace_power.py
    # app_root = parents[4] (coint4/), repo_root = parents[5]
    return Path(__file__).resolve().parents[5]


def _possible_secret_paths() -> list[Path]:
    repo_root = _repo_root()
    app_root = repo_root / "coint4"
    return [
        Path.home() / ".serverspace_api_key",
        Path("/etc/serverspace_api_key"),
        repo_root / ".secrets" / "serverspace_api_key",
        app_root / ".secrets" / "serverspace_api_key",
    ]


def load_api_key() -> str:
    for key_name in ("SERVSPACE_API_KEY", "SERVERSPACE_API_KEY"):
        value = str(os.environ.get(key_name) or "").strip()
        if value:
            return value

    for path in _possible_secret_paths():
        if not path.exists():
            continue
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value

    raise ServerspaceError(
        "Serverspace API key не найден. "
        "Установите SERVSPACE_API_KEY (legacy SERVERSPACE_API_KEY) или создайте файл "
        "~/.serverspace_api_key или /etc/serverspace_api_key "
        "(legacy: .secrets/serverspace_api_key)."
    )


def _api_base(api_base: str | None = None) -> str:
    explicit = str(api_base or "").strip()
    if explicit:
        return explicit.rstrip("/")
    for key_name in ("SERVSPACE_API_BASE", "SERVERSPACE_API_BASE"):
        value = str(os.environ.get(key_name) or "").strip()
        if value:
            return value.rstrip("/")
    return "https://api.serverspace.ru/api/v1"


def _request_json(
    *,
    method: str,
    url: str,
    api_key: str,
    body: dict[str, Any] | None = None,
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
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise ServerspaceError(
            f"HTTP {exc.code} for {method.upper()} {url}: {payload[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ServerspaceError(f"Request failed for {method.upper()} {url}: {exc}") from exc

    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _extract_servers(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "items", "servers"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _iter_server_ips(server: dict[str, Any]) -> Iterable[str]:
    scalar_keys = (
        "ip",
        "ipv4",
        "ip_address",
        "address",
        "public_ip",
        "publicIp",
        "public_ip_address",
        "publicIpAddress",
    )
    for key in scalar_keys:
        value = server.get(key)
        if isinstance(value, str) and value:
            yield value

    nested_keys = ("addresses", "networks", "ips", "ipAddresses", "nics", "interfaces")
    for key in nested_keys:
        value = server.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item:
                    yield item
                if isinstance(item, dict):
                    for nested in item.values():
                        if isinstance(nested, str) and nested:
                            yield nested
        if isinstance(value, dict):
            for nested in value.values():
                if isinstance(nested, str) and nested:
                    yield nested


def _server_id(server: dict[str, Any]) -> str:
    for key in ("id", "uuid", "server_id", "serverId"):
        value = server.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def list_servers(api_key: str, api_base: str | None = None) -> list[dict[str, Any]]:
    payload = _request_json(
        method="GET",
        url=f"{_api_base(api_base)}/servers",
        api_key=api_key,
    )
    return _extract_servers(payload)


def resolve_server_id_by_ip(api_key: str, ip: str, api_base: str | None = None) -> str:
    target_ip = str(ip or "").strip()
    if not target_ip:
        raise ServerspaceError("target ip is empty")
    for server in list_servers(api_key=api_key, api_base=api_base):
        if target_ip not in set(_iter_server_ips(server)):
            continue
        server_id = _server_id(server)
        if server_id:
            return server_id
    raise ServerspaceError(f"Server with IP {target_ip} not found")


def get_status(api_key: str, server_id: str, api_base: str | None = None) -> str:
    sid = str(server_id or "").strip()
    if not sid:
        raise ServerspaceError("server_id is empty")
    for server in list_servers(api_key=api_key, api_base=api_base):
        if _server_id(server) != sid:
            continue
        for key in ("state", "status", "power_state", "powerState"):
            value = server.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return "unknown"
    raise ServerspaceError(f"Server id {sid} not found in list_servers")


def _post_with_tolerated_errors(
    *,
    api_key: str,
    url: str,
    body: dict[str, Any],
    tolerated_substrings: tuple[str, ...],
) -> None:
    try:
        _request_json(method="POST", url=url, api_key=api_key, body=body)
    except ServerspaceError as exc:
        text = str(exc).lower()
        if any(fragment in text for fragment in tolerated_substrings):
            return
        raise


def power_on(api_key: str, server_id: str, api_base: str | None = None) -> None:
    sid = str(server_id or "").strip()
    if not sid:
        raise ServerspaceError("server_id is empty")
    base = _api_base(api_base)
    _post_with_tolerated_errors(
        api_key=api_key,
        url=f"{base}/servers/{sid}/power/on",
        body={"server_id": sid},
        tolerated_substrings=(
            "already on",
            "already started",
            "already running",
            "conflict occurred during the competitive change of the object",
            "wait until the end of the previous operation and try again",
        ),
    )


def power_off(api_key: str, server_id: str, api_base: str | None = None) -> None:
    sid = str(server_id or "").strip()
    if not sid:
        raise ServerspaceError("server_id is empty")
    base = _api_base(api_base)
    _post_with_tolerated_errors(
        api_key=api_key,
        url=f"{base}/servers/{sid}/power/shutdown",
        body={"server_id": sid},
        tolerated_substrings=(
            "already off",
            "already stopped",
            "already shutdown",
            "conflict occurred during the competitive change of the object",
            "wait until the end of the previous operation and try again",
        ),
    )
