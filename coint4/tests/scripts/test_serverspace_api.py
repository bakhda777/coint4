import importlib.util
import sys
from pathlib import Path

import pytest


def _load_serverspace_module():
    app_root = Path(__file__).resolve().parents[2]
    script_path = app_root / "scripts/vps/serverspace_api.py"
    spec = importlib.util.spec_from_file_location("serverspace_api_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_find_server_by_ip_supports_nics_ip_address(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_serverspace_module()
    payload = {
        "data": [
            {
                "id": "srv-123",
                "name": "target-vps",
                "state": "running",
                "nics": [{"ip_address": "85.198.90.128"}],
            }
        ]
    }
    monkeypatch.setattr(module, "_request_json", lambda **_kwargs: payload)

    match = module.find_server_by_ip(
        api_base="https://api.serverspace.ru/api/v1",
        api_key="dummy",
        ip="85.198.90.128",
    )

    assert match is not None
    assert match.server_id == "srv-123"
    assert match.name == "target-vps"
    assert match.state == "running"


def test_load_api_key_reads_secret_file_when_env_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_serverspace_module()
    key_file = tmp_path / ".secrets" / "serverspace_api_key"
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_text("  from-file-key \n", encoding="utf-8")

    monkeypatch.delenv("SERVSPACE_API_KEY", raising=False)
    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)

    assert module._load_api_key() == "from-file-key"  # noqa: SLF001


def test_load_api_key_rejects_empty_secret_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_serverspace_module()
    key_file = tmp_path / ".secrets" / "serverspace_api_key"
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_text(" \n\t", encoding="utf-8")

    monkeypatch.delenv("SERVSPACE_API_KEY", raising=False)
    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="empty"):
        module._load_api_key()  # noqa: SLF001


def test_shutdown_is_idempotent_when_server_already_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_serverspace_module()

    def _raise_already_stopped(**_kwargs):
        raise RuntimeError("HTTP 400 for POST ...: The server is already stopped")

    monkeypatch.setattr(module, "_request_json", _raise_already_stopped)

    module.shutdown(
        api_base="https://api.serverspace.ru/api/v1",
        api_key="dummy",
        server_id="srv-123",
    )
