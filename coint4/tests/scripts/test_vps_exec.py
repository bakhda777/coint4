import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_vps_exec_module() -> Any:
    app_root = Path(__file__).resolve().parents[2]
    script_path = app_root / "scripts/vps/vps_exec.py"
    spec = importlib.util.spec_from_file_location("vps_exec_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ssh_exec_wraps_remote_command_in_bash_lc(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_vps_exec_module()
    captured: dict[str, Any] = {}

    def _fake_run(cmd: list[str], *, text: bool, check: bool) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["text"] = text
        captured["check"] = check
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    target = module.SSHTarget(user="root", host="85.198.90.128", key_path="/tmp/key")

    module.ssh_exec(target, "echo ok", check=False)

    assert captured["cmd"][-1] == "bash -lc 'echo ok'"
    assert captured["text"] is True
    assert captured["check"] is False


def test_tmux_exec_create_true_uses_single_session_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_vps_exec_module()
    captured: dict[str, Any] = {}

    def _fake_ssh_exec(target: Any, cmd: str, *, check: bool) -> subprocess.CompletedProcess[str]:
        captured["target"] = target
        captured["cmd"] = cmd
        captured["check"] = check
        return subprocess.CompletedProcess(["ssh"], 0)

    monkeypatch.setattr(module, "ssh_exec", _fake_ssh_exec)
    target = module.SSHTarget(user="root", host="85.198.90.128", key_path="/tmp/key")

    module.tmux_exec(target, "coint4-wfa", "echo ok", create=True)

    q_session = shlex.quote("coint4-wfa")
    q_cmd = shlex.quote(f"bash -lc {shlex.quote('echo ok')}")
    expected = (
        f"tmux has-session -t {q_session} 2>/dev/null || "
        f"tmux new-session -d -s {q_session}; "
        f"tmux send-keys -t {q_session} {q_cmd} C-m"
    )
    assert captured["cmd"] == expected
    assert captured["check"] is True


def test_tmux_exec_create_false_sends_keys_only(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_vps_exec_module()
    captured: dict[str, Any] = {}

    def _fake_ssh_exec(_target: Any, cmd: str, *, check: bool) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["check"] = check
        return subprocess.CompletedProcess(["ssh"], 0)

    monkeypatch.setattr(module, "ssh_exec", _fake_ssh_exec)
    target = module.SSHTarget(user="root", host="85.198.90.128", key_path="/tmp/key")

    module.tmux_exec(target, "coint4-wfa", "echo ok", create=False)

    q_session = shlex.quote("coint4-wfa")
    q_cmd = shlex.quote(f"bash -lc {shlex.quote('echo ok')}")
    assert captured["cmd"] == f"tmux send-keys -t {q_session} {q_cmd} C-m"
    assert captured["check"] is True


def test_main_exec_accepts_remainder_after_double_dash(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_vps_exec_module()
    captured: dict[str, Any] = {}

    def _fake_ssh_exec(_target: Any, cmd: str, *, check: bool) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["check"] = check
        return subprocess.CompletedProcess(["ssh"], 7)

    monkeypatch.setattr(module, "ssh_exec", _fake_ssh_exec)

    rc = module.main(["exec", "--", "echo", "ok"])

    assert rc == 7
    assert captured["cmd"] == "echo ok"
    assert captured["check"] is False


def test_main_tmux_has_exit_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_vps_exec_module()

    monkeypatch.setattr(module, "tmux_has_session", lambda *_args, **_kwargs: True)
    assert module.main(["tmux-has", "--session", "coint4-wfa"]) == 0

    monkeypatch.setattr(module, "tmux_has_session", lambda *_args, **_kwargs: False)
    assert module.main(["tmux-has", "--session", "coint4-wfa"]) == 2


def test_main_tmux_exec_no_create_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_vps_exec_module()
    captured: dict[str, Any] = {}

    def _fake_tmux_exec(_target: Any, session: str, cmd: str, *, create: bool) -> None:
        captured["session"] = session
        captured["cmd"] = cmd
        captured["create"] = create

    monkeypatch.setattr(module, "tmux_exec", _fake_tmux_exec)

    rc = module.main(["tmux-exec", "--session", "coint4-wfa", "--no-create", "--", "echo", "ok"])

    assert rc == 0
    assert captured["session"] == "coint4-wfa"
    assert captured["cmd"] == "echo ok"
    assert captured["create"] is False


def test_main_exec_rejects_empty_command_after_double_dash() -> None:
    module = _load_vps_exec_module()
    with pytest.raises(SystemExit, match="Missing command"):
        module.main(["exec", "--"])
