from __future__ import annotations

import argparse
import io
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import pytest


def _load_powered_module(tmp_path: Path):
    script_path = Path(__file__).resolve().parents[2] / "scripts/optimization/run_wfa_queue_powered.py"
    spec = importlib.util.spec_from_file_location(f"run_wfa_queue_powered_test_{tmp_path.name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_import_does_not_pull_numba_kernels() -> None:
    before = set(sys.modules)
    _ = _load_powered_module(Path("tmp"))
    after = set(sys.modules)
    added = {name for name in after - before if name.startswith("coint2")}
    assert not any(name.startswith("coint2.core.numba_kernels") for name in added)


@pytest.fixture
def sample_queue(tmp_path: Path) -> Path:
    queue = tmp_path / "artifacts/wfa/aggregate/group/run_queue_mini.csv"
    queue.parent.mkdir(parents=True)
    queue.write_text(
        textwrap.dedent(
            """
            run_name,config_path,status
            run1,configs/a.yaml,planned
            run2,configs/b.yaml,stalled
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return queue


def test_detect_remote_repo_chooses_first_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_powered_module(tmp_path)

    log = []

    def _check(*args, **_kwargs) -> int:
        command = str(_kwargs.get("remote_command", ""))
        if not command and len(args) > 2:
            command = str(args[2])
        if "/good/path" in command:
            return 0
        return 1

    monkeypatch.setattr(module, "_run_remote_command", _check)

    host = "85.198.90.128"
    repo = module._detect_remote_repo(
        host=host,
        user="root",
        candidates=[Path("/bad/path"), Path("/good/path")],
        port=22,
        log_path=tmp_path / "log.log",
        log=lambda msg: log.append(msg),
    )

    assert str(repo) == "/good/path"
    assert "remote_repo_check=/bad/path" in log[0]
    assert "remote_repo_check=/good/path" in log[1]
    assert "remote_repo_detected=/good/path" in log[-1]


def test_sync_inputs_uploads_queue_and_configs(
    monkeypatch: pytest.MonkeyPatch, sample_queue: Path, tmp_path: Path
) -> None:
    module = _load_powered_module(sample_queue.parent)
    project_root = tmp_path

    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    cfg_a = project_root / "configs/a.yaml"
    cfg_b = project_root / "configs/b.yaml"
    cfg_a.write_text("a: 1\n", encoding="utf-8")
    cfg_b.write_text("b: 2\n", encoding="utf-8")

    mk_mkdir_calls: list[str] = []
    scp_calls: list[str] = []

    monkeypatch.setattr(module, "_run_remote_command", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        module,
        "_run_scp_file",
        lambda source, destination_user, destination_host, destination, *, port, log: scp_calls.append(f"{source}->{destination_user}@{destination_host}:{destination}"),
    )

    monkeypatch.setattr(module, "_run_remote_mkdir", lambda *args, **kwargs: mk_mkdir_calls.append(str(args[2])))

    module._sync_inputs(
        host="85.198.90.128",
        user="root",
        queue_path=sample_queue,
        remote_repo=Path("/remote/repo"),
        remote_python="python3",
        project_root=project_root,
        bulk_configs=False,
        force_remote_queue_overwrite=True,
        port=22,
        log_path=tmp_path / "log.log",
        log=lambda msg: None,
    )

    assert any("run_queue_mini.csv" in item for item in scp_calls)
    assert any(str(project_root / "configs/a.yaml") in item for item in scp_calls)
    assert any(str(project_root / "configs/b.yaml") in item for item in scp_calls)
    assert len(scp_calls) == 3
    assert len(mk_mkdir_calls) == 3


def test_sync_inputs_skips_queue_upload_when_remote_has_progress(
    monkeypatch: pytest.MonkeyPatch, sample_queue: Path, tmp_path: Path
) -> None:
    module = _load_powered_module(sample_queue.parent)
    project_root = tmp_path
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    (project_root / "configs/a.yaml").write_text("a: 1\n", encoding="utf-8")
    (project_root / "configs/b.yaml").write_text("b: 2\n", encoding="utf-8")

    scp_calls: list[str] = []
    fetch_calls: list[str] = []
    monkeypatch.setattr(module, "_run_remote_command", lambda *args, **kwargs: 0)
    monkeypatch.setattr(module, "_run_remote_mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_run_scp_file",
        lambda source, destination_user, destination_host, destination, *, port, log: scp_calls.append(
            f"{source}->{destination_user}@{destination_host}:{destination}"
        ),
    )
    monkeypatch.setattr(
        module,
        "_fetch_remote_file",
        lambda source_user, source_host, source, destination, *, port, log, atomic=False: fetch_calls.append(
            f"{source_user}@{source_host}:{source}->{destination}"
        ),
    )
    monkeypatch.setattr(
        module,
        "get_remote_queue_counts",
        lambda **_kwargs: {
            "ok": True,
            "queue_exists": True,
            "counts": {"completed": 1, "running": 2},
            "total": 3,
            "has_metrics": True,
        },
    )

    module._sync_inputs(
        host="85.198.90.128",
        user="root",
        queue_path=sample_queue,
        remote_repo=Path("/remote/repo"),
        remote_python="python3",
        project_root=project_root,
        bulk_configs=False,
        force_remote_queue_overwrite=False,
        port=22,
        log_path=tmp_path / "log.log",
        log=lambda msg: None,
    )

    assert fetch_calls, "remote queue should be fetched back when progress exists"
    assert all(str(sample_queue) not in call for call in scp_calls), "local queue upload must be skipped"
    assert len(scp_calls) == 2, "only config uploads are expected"


def test_sync_inputs_uploads_queue_when_remote_no_progress(
    monkeypatch: pytest.MonkeyPatch, sample_queue: Path, tmp_path: Path
) -> None:
    module = _load_powered_module(sample_queue.parent)
    project_root = tmp_path
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    (project_root / "configs/a.yaml").write_text("a: 1\n", encoding="utf-8")
    (project_root / "configs/b.yaml").write_text("b: 2\n", encoding="utf-8")

    scp_calls: list[str] = []
    monkeypatch.setattr(module, "_run_remote_command", lambda *args, **kwargs: 0)
    monkeypatch.setattr(module, "_run_remote_mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_run_scp_file",
        lambda source, destination_user, destination_host, destination, *, port, log: scp_calls.append(
            f"{source}->{destination_user}@{destination_host}:{destination}"
        ),
    )
    monkeypatch.setattr(
        module,
        "get_remote_queue_counts",
        lambda **_kwargs: {
            "ok": True,
            "queue_exists": True,
            "counts": {"planned": 10},
            "total": 10,
            "has_metrics": False,
        },
    )

    module._sync_inputs(
        host="85.198.90.128",
        user="root",
        queue_path=sample_queue,
        remote_repo=Path("/remote/repo"),
        remote_python="python3",
        project_root=project_root,
        bulk_configs=False,
        force_remote_queue_overwrite=False,
        port=22,
        log_path=tmp_path / "log.log",
        log=lambda msg: None,
    )

    assert any(str(sample_queue) in call for call in scp_calls), "queue upload should happen without remote progress"
    assert len(scp_calls) == 3


def test_no_retry_when_repo_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_powered_module(tmp_path)

    run_calls: list[str] = []

    def _remote_cmd(*_args, **_kwargs):
        run_calls.append("run")
        return 1, "bash: cd: /no/such/path: No such file or directory"

    monkeypatch.setattr(module, "_exec_ssh_command", lambda *args, **kwargs: _remote_cmd(*args, **kwargs))

    with pytest.raises(module._PoweredFailure) as exc_info:
        module._run_remote_command(
            "host",
            "root",
            "cd /no/such/path && true",
            log_path=tmp_path / "log.log",
            port=22,
            log=lambda msg: None,
            command_purpose="queue-run",
        )
    assert exc_info.value.error_class == "REMOTE_REPO_NOT_FOUND"
    assert exc_info.value.fatal is True
    assert run_calls == ["run"]


def test_dry_run_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    argv = ["--queue", "run_queue.csv", "--dry-run", "--max-retries", "1", "--backoff-seconds", "0"]
    module = _load_powered_module(tmp_path)

    queue = tmp_path / "run_queue.csv"
    queue.write_text("run_name,config_path,status\nrun1,configs/a.yaml,planned\n", encoding="utf-8")

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/a.yaml").write_text("k: v\n", encoding="utf-8")

    monkeypatch.setattr(module, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(module, "_detect_remote_repo", lambda *_, **__: Path("/remote/repo"))
    monkeypatch.setattr(module, "_detect_remote_python", lambda *_, **__: "/bin/python3")
    monkeypatch.setattr(module, "_sync_inputs", lambda *_, **__: None)
    monkeypatch.setattr(module, "_safe_api_key", lambda *_args, **_kwargs: "dummy")
    monkeypatch.setattr(module, "_safe_resolve_server_id_by_ip", lambda *_args, **_kwargs: "srv-1")
    monkeypatch.setattr(module, "ensure_server_ready", lambda *_, **__: None)
    monkeypatch.setattr(module.time, "sleep", lambda *_args, **_kwargs: None)

    def _run_remote_command(host, user, remote_command, **_kwargs):
        if "POWER_OK" in remote_command:
            return 0
        return 1

    monkeypatch.setattr(module, "_run_remote_command", _run_remote_command)
    monkeypatch.setattr(module, "_safe_power_off", lambda *_, **__: None)

    result = module.main(argv)
    captured = capsys.readouterr()
    assert result == 0
    log_path = module._log_file_path(tmp_path, queue)
    assert log_path.exists()
    text = log_path.read_text(encoding="utf-8")
    assert "powered: start dry_run=True" in text
    assert "powered: DRY_RUN_SUCCESS" in text
    assert "DRY_RUN_SUCCESS" in captured.err


def test_detect_remote_repo_after_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_powered_module(tmp_path)
    queue = tmp_path / "run_queue.csv"
    queue.write_text("run_name,config_path,status\nrun1,configs/a.yaml,planned\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/a.yaml").write_text("k: v\n", encoding="utf-8")

    events: list[str] = []

    monkeypatch.setattr(module, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(module, "_safe_api_key", lambda *_args, **_kwargs: "dummy")
    monkeypatch.setattr(module, "_safe_resolve_server_id_by_ip", lambda *_args, **_kwargs: "srv-1")

    def _ensure_server_ready(*_args, **_kwargs) -> None:
        events.append("wait_ssh_ready")

    monkeypatch.setattr(module, "ensure_server_ready", _ensure_server_ready)

    def _run_remote_command(*_args, **_kwargs) -> int:
        events.append("preflight_cmd")
        return 0

    monkeypatch.setattr(module, "_run_remote_command", _run_remote_command)

    monkeypatch.setattr(
        module,
        "_detect_remote_repo",
        lambda *_args, **_kwargs: (events.append("detect_remote_repo"), Path("/remote/repo"))[1],
    )

    monkeypatch.setattr(
        module,
        "_detect_remote_python",
        lambda *_args, **_kwargs: events.append("detect_remote_python") or "/usr/bin/python3",
    )

    def _sync_inputs(*_args, **_kwargs) -> None:
        events.append("sync_inputs")

    monkeypatch.setattr(module, "_sync_inputs", _sync_inputs)
    monkeypatch.setattr(module, "_safe_power_off", lambda *_, **__: None)

    argv = ["--queue", str(queue), "--dry-run", "--max-retries", "1", "--backoff-seconds", "0", "--sync-inputs", "true"]
    result = module.main(argv)
    assert result == 0

    assert events.index("detect_remote_repo") > events.index("wait_ssh_ready")
    assert events.index("detect_remote_repo") > events.index("preflight_cmd")
    assert events.index("detect_remote_python") > events.index("detect_remote_repo")
    assert events.index("sync_inputs") > events.index("detect_remote_python")


def test_detect_remote_repo_after_ready_without_preflight_cmd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_powered_module(tmp_path)
    queue = tmp_path / "run_queue.csv"
    queue.write_text("run_name,config_path,status\nrun1,configs/a.yaml,planned\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/a.yaml").write_text("k: v\n", encoding="utf-8")

    events: list[str] = []

    monkeypatch.setattr(module, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(module, "_safe_api_key", lambda *_args, **_kwargs: "dummy")
    monkeypatch.setattr(module, "_safe_resolve_server_id_by_ip", lambda *_args, **_kwargs: "srv-1")

    def _ensure_server_ready(*_args, **_kwargs) -> None:
        events.append("wait_ssh_ready")

    monkeypatch.setattr(module, "ensure_server_ready", _ensure_server_ready)

    monkeypatch.setattr(
        module,
        "_run_remote_command",
        lambda *args, **kwargs: 0,
    )
    monkeypatch.setattr(
        module,
        "_detect_remote_repo",
        lambda *_, **__: (events.append("detect_remote_repo"), Path("/remote/repo"))[1],
    )
    monkeypatch.setattr(
        module,
        "_detect_remote_python",
        lambda *_, **__: events.append("detect_remote_python") or "/usr/bin/python3",
    )
    monkeypatch.setattr(module, "_sync_inputs", lambda *_, **__: None)
    monkeypatch.setattr(module, "_safe_power_off", lambda *_, **__: None)

    argv = [
        "--queue",
        str(queue),
        "--dry-run",
        "--preflight",
        "false",
        "--max-retries",
        "1",
        "--backoff-seconds",
        "0",
    ]
    result = module.main(argv)
    assert result == 0

    assert events.index("detect_remote_repo") > events.index("wait_ssh_ready")
    assert events.index("detect_remote_python") > events.index("detect_remote_repo")


def test_run_with_retry_backoff_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_powered_module(tmp_path)

    attempts: list[int] = []
    logs: list[str] = []

    def _attempt() -> int:
        attempts.append(1)
        if len(attempts) < 3:
            raise module._PoweredFailure(
                "tmp",
                error_class="NETWORK",
                fatal=False,
            )
        return 0

    monkeypatch.setattr(module.time, "sleep", lambda *_args, **_kwargs: None)
    result = module._run_with_retries(
        _attempt,
        max_retries=3,
        backoff_seconds=10.0,
        log=lambda msg: logs.append(msg),
    )

    assert result == 0
    assert len(attempts) == 3
    assert any("error_class=NETWORK" in m for m in logs)
    assert any("retry:" in m for m in logs)


def test_sync_repo_code_retries_transient_stream_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_powered_module(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/app.py").write_text("value = 1\n", encoding="utf-8")

    local_manifest = module._build_local_sync_manifest(tmp_path, module._sync_code_include_paths(tmp_path))
    manifest_calls: list[str] = []

    def _remote_sync_manifest(**_kwargs):
        manifest_calls.append("manifest")
        if len(manifest_calls) == 1:
            return {"digest": "remote-before", "count": 1}
        return {"digest": local_manifest["digest"], "count": local_manifest["count"]}

    monkeypatch.setattr(module, "_remote_sync_manifest", _remote_sync_manifest)
    monkeypatch.setattr(module, "_git_head_revision", lambda *_args, **_kwargs: "local-head")
    monkeypatch.setattr(module, "_remote_git_head_revision", lambda **_kwargs: "remote-head")
    monkeypatch.setattr(module, "_local_paths_have_tracked_changes", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(module.time, "sleep", lambda *_args, **_kwargs: None)

    class _FakeStdout:
        def close(self) -> None:
            return None

    class _FakeTarProc:
        def __init__(self) -> None:
            self.stdout = _FakeStdout()
            self.stderr = io.BytesIO(b"")

        def wait(self, timeout=None) -> int:
            _ = timeout
            return 0

        def kill(self) -> None:
            return None

    class _FakeSshProc:
        def __init__(self, *, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self._stdout = stdout
            self._stderr = stderr

        def communicate(self, timeout=None) -> tuple[str, str]:
            _ = timeout
            return self._stdout, self._stderr

        def kill(self) -> None:
            return None

    ssh_attempts: list[int] = []

    def _popen(cmd, **_kwargs):
        program = Path(str(cmd[0])).name
        if program == "tar":
            return _FakeTarProc()
        if program == "ssh":
            ssh_attempts.append(1)
            if len(ssh_attempts) == 1:
                return _FakeSshProc(
                    returncode=255,
                    stderr="ssh: connect to host 85.198.90.128 port 22: Connection timed out",
                )
            return _FakeSshProc(returncode=0)
        raise AssertionError(f"unexpected popen program: {cmd}")

    monkeypatch.setattr(module.subprocess, "Popen", _popen)

    logs: list[str] = []
    module._sync_repo_code(
        host="85.198.90.128",
        user="root",
        project_root=tmp_path,
        remote_repo=Path("/remote/repo"),
        port=22,
        log_path=tmp_path / "powered.log",
        log=lambda msg: logs.append(msg),
    )

    assert len(ssh_attempts) == 2
    assert len(manifest_calls) == 2
    assert any("sync_retry description=sync_code_stream" in entry for entry in logs)
    assert any("sync_code ok" in entry for entry in logs)


def test_sync_repo_code_prunes_remote_only_files_before_stream_when_parity_is_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_powered_module(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/app.py").write_text("value = 1\n", encoding="utf-8")

    local_manifest = module._build_local_sync_manifest(tmp_path, module._sync_code_include_paths(tmp_path))
    local_entries = list(local_manifest["entries"])
    remote_before = {
        "digest": "remote-before",
        "count": int(local_manifest["count"]) + 1,
        "entries": local_entries
        + [{"path": "scripts/data/build_market_metrics.py", "sha256": "deadbeef", "size": 17}],
    }
    manifest_payloads = [remote_before, local_manifest]
    pruned_paths: list[str] = []

    monkeypatch.setattr(
        module,
        "_remote_sync_manifest",
        lambda **_kwargs: manifest_payloads.pop(0),
    )
    monkeypatch.setattr(
        module,
        "_prune_remote_sync_extras",
        lambda **kwargs: pruned_paths.extend(list(kwargs.get("extra_paths") or [])) or len(pruned_paths),
    )
    monkeypatch.setattr(module, "_git_head_revision", lambda *_args, **_kwargs: "local-head")
    monkeypatch.setattr(module, "_remote_git_head_revision", lambda **_kwargs: "remote-head")
    monkeypatch.setattr(module, "_local_paths_have_tracked_changes", lambda *_args, **_kwargs: False)

    def _unexpected_popen(*_args, **_kwargs):
        raise AssertionError("stream sync should be skipped once stale remote-only files are pruned")

    monkeypatch.setattr(module.subprocess, "Popen", _unexpected_popen)

    logs: list[str] = []
    module._sync_repo_code(
        host="85.198.90.128",
        user="root",
        project_root=tmp_path,
        remote_repo=Path("/remote/repo"),
        port=22,
        log_path=tmp_path / "powered.log",
        log=lambda msg: logs.append(msg),
    )

    assert pruned_paths == ["scripts/data/build_market_metrics.py"]
    assert any("remote_only_before count=1" in entry for entry in logs)
    assert any("reason=post_prune_manifest_match" in entry for entry in logs)


def test_wait_for_completion_until_metrics_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_powered_module(tmp_path)
    attempts: list[int] = []
    sync_calls: list[str] = []

    probe_payloads = [
        {
            "ok": True,
            "total": 4,
            "counts": {"running": 4},
            "all_done": False,
            "has_metrics": False,
        },
        {
            "ok": True,
            "total": 4,
            "counts": {"completed": 4},
            "all_done": True,
            "has_metrics": True,
        },
    ]

    def _exec(*_args, **_kwargs):
        attempts.append(1)
        payload = probe_payloads.pop(0)
        return 0, json.dumps(payload, sort_keys=True) + "\n"

    monkeypatch.setattr(module, "_exec_ssh_command", _exec)
    monkeypatch.setattr(
        module,
        "_sync_active_remote_queue_back",
        lambda **kwargs: sync_calls.append(str(kwargs.get("queue_relative") or "")),
    )
    monkeypatch.setattr(module.time, "sleep", lambda *_args, **_kwargs: None)

    logs: list[str] = []
    module._wait_for_completion(
        host="85.198.90.128",
        user="root",
        remote_repo=Path("/remote/repo"),
        remote_python="python3",
        queue_relative="artifacts/wfa/aggregate/group/run_queue_mini.csv",
        log_path=tmp_path / "powered.log",
        port=22,
        timeout_sec=300,
        poll_sec=1,
        local_queue_path=tmp_path / "run_queue.csv",
        log=lambda msg: logs.append(msg),
    )

    assert len(attempts) == 2
    assert sync_calls == [
        "artifacts/wfa/aggregate/group/run_queue_mini.csv",
        "artifacts/wfa/aggregate/group/run_queue_mini.csv",
    ]
    assert any("wait_completion attempt=1" in line for line in logs)
    assert any("completion detected" in line for line in logs)


def test_get_remote_queue_counts_returns_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_powered_module(tmp_path)
    payload = {
        "ok": True,
        "queue": "/remote/repo/artifacts/wfa/aggregate/group/run_queue.csv",
        "total": 6,
        "counts": {"planned": 1, "running": 4, "completed": 1},
        "all_done": False,
        "has_metrics": True,
    }

    monkeypatch.setattr(
        module,
        "_exec_ssh_command",
        lambda *_args, **_kwargs: (0, json.dumps(payload, sort_keys=True) + "\n"),
    )

    result = module.get_remote_queue_counts(
        host="85.198.90.128",
        user="root",
        remote_repo=Path("/remote/repo"),
        remote_python="python3",
        queue_relative="artifacts/wfa/aggregate/group/run_queue.csv",
        port=22,
        log_path=tmp_path / "probe.log",
        log=lambda _msg: None,
    )

    assert result["ok"] is True
    assert result["total"] == 6
    assert result["counts"]["running"] == 4
    assert result["has_metrics"] is True


def test_remote_rebuild_rollup_rebuilds_full_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_powered_module(tmp_path)
    captured: dict[str, str] = {}

    def _fake_run_remote_command(host, user, remote_command, **_kwargs):
        _ = (host, user)
        captured["cmd"] = str(remote_command)
        return 0

    monkeypatch.setattr(module, "_run_remote_command", _fake_run_remote_command)

    module._remote_rebuild_rollup(
        host="85.198.90.128",
        user="root",
        remote_repo=Path("/remote/repo"),
        remote_python="python3",
        queue_relative="artifacts/wfa/aggregate/group/run_queue.csv",
        log_path=tmp_path / "powered.log",
        port=22,
        log=lambda _msg: None,
    )

    cmd = captured.get("cmd", "")
    assert "build_run_index.py --output-dir artifacts/wfa/aggregate/rollup" in cmd
    assert "--queue" not in cmd


def test_get_remote_queue_counts_maps_queue_missing_error_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_powered_module(tmp_path)
    payload = {"ok": False, "error": "QUEUE_MISSING", "queue": "/remote/repo/queue.csv"}
    monkeypatch.setattr(
        module,
        "_exec_ssh_command",
        lambda *_args, **_kwargs: (2, json.dumps(payload, ensure_ascii=False) + "\n"),
    )

    with pytest.raises(module._PoweredFailure) as exc_info:
        module.get_remote_queue_counts(
            host="85.198.90.128",
            user="root",
            remote_repo=Path("/remote/repo"),
            remote_python="python3",
            queue_relative="artifacts/wfa/aggregate/group/run_queue.csv",
            port=22,
            log_path=tmp_path / "probe.log",
            log=lambda _msg: None,
        )

    assert exc_info.value.error_class == "QUEUE_MISSING"
    assert exc_info.value.fatal is True


def test_run_powered_queue_waits_after_queue_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_powered_module(tmp_path)
    queue = tmp_path / "run_queue.csv"
    queue.write_text("run_name,config_path,status\nrun1,configs/a.yaml,planned\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/a.yaml").write_text("k: v\n", encoding="utf-8")

    events: list[str] = []

    monkeypatch.setattr(module, "_safe_api_key", lambda *_args, **_kwargs: "dummy")
    monkeypatch.setattr(module, "_safe_resolve_server_id_by_ip", lambda *_args, **_kwargs: "srv-1")
    monkeypatch.setattr(module, "ensure_server_ready", lambda *_args, **_kwargs: events.append("ready"))
    monkeypatch.setattr(module, "_safe_power_off", lambda *_args, **_kwargs: events.append("poweroff"))
    monkeypatch.setattr(module, "_resolve_remote_repo", lambda *_args, **_kwargs: events.append("resolve_repo") or Path("/remote/repo"))
    monkeypatch.setattr(module, "_detect_remote_python", lambda *_args, **_kwargs: events.append("detect_python") or "python3")
    monkeypatch.setattr(module, "_sync_inputs", lambda *_args, **_kwargs: events.append("sync_inputs"))

    def _remote_cmd(*_args, **_kwargs):
        purpose = str(_kwargs.get("command_purpose") or "")
        if purpose:
            events.append(purpose)
        return 0

    monkeypatch.setattr(module, "_run_remote_command", _remote_cmd)
    monkeypatch.setattr(
        module,
        "_wait_for_completion",
        lambda *_args, **_kwargs: events.append("wait_completion"),
    )
    monkeypatch.setattr(
        module,
        "_remote_rebuild_rollup",
        lambda *_args, **_kwargs: events.append("remote_rollup"),
    )
    monkeypatch.setattr(
        module,
        "_sync_rollup_back",
        lambda *_args, **_kwargs: events.append("sync_rollup_back"),
    )
    monkeypatch.setattr(
        module,
        "_remote_rank_and_sync",
        lambda *_args, **_kwargs: events.append("remote_rank") or (tmp_path / "rank_result.json"),
    )

    args = argparse.Namespace(
        queue=str(queue),
        compute_host="85.198.90.128",
        serverspace_server_id=None,
        ssh_user="root",
        ssh_port=22,
        preflight=True,
        remote_repo="auto",
        remote_repo_candidates=["/opt/coint4/coint4"],
        bootstrap_repo=False,
        bootstrap_remote_dir="/opt/coint4",
        bootstrap_venv=False,
        sync_inputs=True,
        sync_configs_bulk=False,
        force_remote_queue_overwrite=False,
        probe_queue=False,
        dry_run=False,
        statuses="planned,stalled",
        parallel=1,
        postprocess=True,
        max_retries=1,
        backoff_seconds=0.0,
        poweroff=True,
        wait_completion=True,
        wait_timeout_sec=300,
        wait_poll_sec=1,
    )

    rc = module.run_powered_queue(args, project_root=tmp_path, log_path=tmp_path / "powered.log")
    assert rc == 0
    assert "queue-run" in events
    assert "wait_completion" in events
    assert events.index("wait_completion") > events.index("queue-run")
    assert "remote-postprocess-queue" in events
    assert events.index("remote-postprocess-queue") > events.index("wait_completion")
    assert events.index("sync_rollup_back") > events.index("remote-postprocess-queue")
    assert events.index("remote_rank") > events.index("sync_rollup_back")


def test_run_powered_queue_logs_degraded_success_when_sync_back_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_powered_module(tmp_path)
    queue = tmp_path / "run_queue.csv"
    queue.write_text("run_name,config_path,status\nrun1,configs/a.yaml,planned\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/a.yaml").write_text("k: v\n", encoding="utf-8")

    monkeypatch.setattr(module, "_safe_api_key", lambda *_args, **_kwargs: "dummy")
    monkeypatch.setattr(module, "_safe_resolve_server_id_by_ip", lambda *_args, **_kwargs: "srv-1")
    monkeypatch.setattr(module, "ensure_server_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_safe_power_off", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_resolve_remote_repo", lambda *_args, **_kwargs: Path("/remote/repo"))
    monkeypatch.setattr(module, "_detect_remote_python", lambda *_args, **_kwargs: "python3")
    monkeypatch.setattr(module, "_sync_inputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_run_remote_command", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        module,
        "_sync_rollup_back",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            module._PoweredFailure("sync-back", error_class="NETWORK", fatal=False)
        ),
    )
    monkeypatch.setattr(module, "_fetch_stalled_diagnostics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_remote_rank_and_sync", lambda *_args, **_kwargs: tmp_path / "rank.json")

    args = argparse.Namespace(
        queue=str(queue),
        compute_host="85.198.90.128",
        serverspace_server_id=None,
        ssh_user="root",
        ssh_port=22,
        preflight=False,
        remote_repo="auto",
        remote_repo_candidates=["/opt/coint4/coint4"],
        bootstrap_repo=False,
        bootstrap_remote_dir="/opt/coint4",
        bootstrap_venv=False,
        sync_inputs=False,
        sync_configs_bulk=False,
        force_remote_queue_overwrite=False,
        probe_queue=False,
        dry_run=False,
        statuses="planned",
        parallel=1,
        postprocess=False,
        max_retries=1,
        backoff_seconds=0.0,
        poweroff=True,
        wait_completion=False,
        wait_timeout_sec=300,
        wait_poll_sec=1,
        skip_power=False,
        watchdog=False,
        watchdog_stale_sec=900,
        sync_code=False,
        cleanup_remote_runs=False,
    )

    rc = module.run_powered_queue(args, project_root=tmp_path, log_path=tmp_path / "powered.log")
    assert rc == 0
    log_text = (tmp_path / "powered.log").read_text(encoding="utf-8")
    assert "powered: degraded_success" in log_text
    assert "sync_back:NETWORK" in log_text


def test_run_powered_queue_skip_power_uses_ssh_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_powered_module(tmp_path)
    queue = tmp_path / "run_queue.csv"
    queue.write_text("run_name,config_path,status\nrun1,configs/a.yaml,planned\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/a.yaml").write_text("k: v\n", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "_safe_api_key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("_safe_api_key must not be used")),
    )
    monkeypatch.setattr(
        module,
        "_safe_resolve_server_id_by_ip",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("server_id API must not be used")),
    )
    monkeypatch.setattr(
        module,
        "ensure_server_ready",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ensure_server_ready must not be used")),
    )
    monkeypatch.setattr(module, "_resolve_remote_repo", lambda *_args, **_kwargs: Path("/remote/repo"))
    monkeypatch.setattr(module, "_detect_remote_python", lambda *_args, **_kwargs: "python3")
    monkeypatch.setattr(module, "_sync_inputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_safe_power_off",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("power_off must not be used in skip_power")),
    )

    purposes: list[str] = []

    def _remote_cmd(_host, _user, _remote_command, *args, **kwargs):
        purposes.append(str(kwargs.get("command_purpose") or ""))
        return 0

    monkeypatch.setattr(module, "_run_remote_command", _remote_cmd)

    args = argparse.Namespace(
        queue=str(queue),
        compute_host="85.198.90.128",
        serverspace_server_id=None,
        ssh_user="root",
        ssh_port=22,
        preflight=False,
        remote_repo="auto",
        remote_repo_candidates=["/opt/coint4/coint4"],
        bootstrap_repo=False,
        bootstrap_remote_dir="/opt/coint4",
        bootstrap_venv=False,
        sync_inputs=False,
        sync_configs_bulk=False,
        force_remote_queue_overwrite=False,
        probe_queue=False,
        dry_run=True,
        statuses="planned,stalled",
        parallel=1,
        postprocess=True,
        max_retries=1,
        backoff_seconds=0.0,
        poweroff=True,
        skip_power=True,
        wait_completion=False,
        wait_timeout_sec=300,
        wait_poll_sec=1,
        watchdog=False,
        watchdog_stale_sec=900,
    )

    rc = module.run_powered_queue(args, project_root=tmp_path, log_path=tmp_path / "powered.log")
    assert rc == 0
    assert "skip-power-preflight" in purposes


def test_parallel_auto_detects_nproc_and_pins_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_powered_module(tmp_path)
    queue = tmp_path / "run_queue.csv"
    queue.write_text("run_name,config_path,status\nrun1,configs/a.yaml,planned\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/a.yaml").write_text("k: v\n", encoding="utf-8")

    monkeypatch.setattr(module, "_safe_api_key", lambda *_args, **_kwargs: "dummy")
    monkeypatch.setattr(module, "_safe_resolve_server_id_by_ip", lambda *_args, **_kwargs: "srv-1")
    monkeypatch.setattr(module, "ensure_server_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_safe_power_off", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_resolve_remote_repo", lambda *_args, **_kwargs: Path("/remote/repo"))
    monkeypatch.setattr(module, "_detect_remote_python", lambda *_args, **_kwargs: "python3")
    monkeypatch.setattr(module, "_sync_inputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_remote_rebuild_rollup", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_sync_rollup_back", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_remote_rank_and_sync", lambda *_args, **_kwargs: tmp_path / "rank.json")
    monkeypatch.setattr(module, "_run_with_retries", lambda fn, **_kwargs: fn())

    commands: dict[str, str] = {}

    def _remote_cmd(_host, _user, remote_command, *args, **kwargs):
        purpose = str(kwargs.get("command_purpose") or "")
        if purpose:
            commands[purpose] = str(remote_command)
        return 0

    monkeypatch.setattr(module, "_run_remote_command", _remote_cmd)
    monkeypatch.setattr(module, "_exec_ssh_command", lambda *_args, **_kwargs: (0, "8\n"))

    args = argparse.Namespace(
        queue=str(queue),
        compute_host="85.198.90.128",
        serverspace_server_id=None,
        ssh_user="root",
        ssh_port=22,
        preflight=True,
        remote_repo="auto",
        remote_repo_candidates=["/opt/coint4/coint4"],
        bootstrap_repo=False,
        bootstrap_remote_dir="/opt/coint4",
        bootstrap_venv=False,
        sync_inputs=False,
        force_remote_queue_overwrite=False,
        sync_configs_bulk=False,
        probe_queue=False,
        dry_run=False,
        statuses="planned,stalled",
        parallel="auto",
        postprocess=True,
        max_retries=1,
        backoff_seconds=0.0,
        poweroff=True,
        wait_completion=False,
        wait_timeout_sec=300,
        wait_poll_sec=1,
    )

    log_path = tmp_path / "powered.log"
    rc = module.run_powered_queue(args, project_root=tmp_path, log_path=log_path)
    assert rc == 0
    queue_cmd = commands.get("queue-run", "")
    assert "--parallel 8" in queue_cmd
    assert "export OMP_NUM_THREADS=1;" in queue_cmd
    assert "export MKL_NUM_THREADS=1;" in queue_cmd
    assert "export OPENBLAS_NUM_THREADS=1;" in queue_cmd
    assert "export NUMBA_NUM_THREADS=1;" in queue_cmd
    assert "export NUMEXPR_NUM_THREADS=1;" in queue_cmd
    log_text = log_path.read_text(encoding="utf-8")
    assert "parallel auto detected nproc=8 chosen_parallel=8" in log_text
    assert "powered: remote_env pinned_threads=1" in log_text


def test_bootstrap_repo_when_remote_repo_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_powered_module(tmp_path)

    queue = tmp_path / "run_queue.csv"
    queue.write_text("run_name,config_path,status\nrun1,configs/a.yaml,planned\n", encoding="utf-8")

    monkeypatch.setattr(module, "_safe_api_key", lambda *_args, **_kwargs: "dummy")
    monkeypatch.setattr(module, "_safe_resolve_server_id_by_ip", lambda *_args, **_kwargs: "srv-1")
    monkeypatch.setattr(module, "ensure_server_ready", lambda *_, **__: None)
    monkeypatch.setattr(module, "_safe_power_off", lambda *_, **__: None)

    def _raise_repo_not_found(*_args, **_kwargs):
        raise module._PoweredFailure("missing", error_class="REMOTE_REPO_NOT_FOUND", fatal=True)

    monkeypatch.setattr(module, "_resolve_remote_repo", _raise_repo_not_found)

    bundle_calls: list[str] = []
    monkeypatch.setattr(
        module,
        "_build_local_repo_bundle",
        lambda *, project_root, bundle_path, log: bundle_calls.append(str(bundle_path)),
    )

    mkdir_calls: list[str] = []
    monkeypatch.setattr(module, "_run_remote_mkdir", lambda *args, **kwargs: mkdir_calls.append(str(args[2])))

    scp_calls: list[str] = []
    monkeypatch.setattr(
        module,
        "_run_scp_file",
        lambda source, destination_user, destination_host, destination, *, port, log: scp_calls.append(
            f"{source}->{destination_user}@{destination_host}:{destination}"
        ),
    )

    remote_cmd_calls: list[tuple[str, str]] = []

    def _remote_cmd(_host, _user, remote_command, *args, **kwargs):
        command_purpose = str(kwargs.get("command_purpose") or "")
        remote_cmd_calls.append((command_purpose, remote_command))
        return 0

    monkeypatch.setattr(module, "_run_remote_command", _remote_cmd)
    monkeypatch.setattr(module, "_detect_remote_repo", lambda *_, **__: Path("/opt/coint4/coint4"))

    detected_repo: list[str] = []
    monkeypatch.setattr(
        module,
        "_detect_remote_python",
        lambda _host, _user, remote_repo, **_kwargs: detected_repo.append(str(remote_repo))
        or "/opt/coint4/coint4/.venv/bin/python",
    )

    args = argparse.Namespace(
        queue=str(queue),
        compute_host="85.198.90.128",
        serverspace_server_id=None,
        ssh_user="root",
        ssh_port=22,
        preflight=False,
        remote_repo="auto",
        remote_repo_candidates=["/missing/one", "/missing/two"],
        bootstrap_repo=True,
        bootstrap_remote_dir="/opt/coint4",
        bootstrap_venv=False,
        sync_inputs=False,
        sync_configs_bulk=False,
        force_remote_queue_overwrite=False,
        probe_queue=False,
        dry_run=True,
        statuses="planned,stalled",
        parallel=1,
        postprocess=True,
        max_retries=1,
        backoff_seconds=0.0,
        poweroff=True,
        wait_completion=True,
        wait_timeout_sec=300,
        wait_poll_sec=1,
    )

    rc = module.run_powered_queue(args, project_root=tmp_path, log_path=tmp_path / "powered.log")
    assert rc == 0
    assert bundle_calls
    assert any("coint4_repo_bundle.tar.gz" in item for item in scp_calls)
    assert any(purpose == "bootstrap-repo" and "tar -xzf" in cmd for purpose, cmd in remote_cmd_calls)
    assert detected_repo == ["/opt/coint4/coint4"]
    assert any(path.endswith("/opt/coint4") for path in mkdir_calls)


def test_remote_rank_and_sync_writes_local_payload_when_run_index_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_powered_module(tmp_path)

    monkeypatch.setattr(module, "_run_remote_command", lambda *args, **kwargs: 0)

    logs: list[str] = []
    local_path = module._remote_rank_and_sync(
        host="85.198.90.128",
        user="root",
        remote_repo=Path("/remote/repo"),
        remote_python="python3",
        queue_relative="artifacts/wfa/aggregate/group/run_queue.csv",
        run_group="group",
        project_root=tmp_path,
        log_path=tmp_path / "powered.log",
        port=22,
        log=lambda msg: logs.append(msg),
    )

    assert local_path.exists()
    payload = json.loads(local_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["details"] == "RUN_INDEX_MISSING_LOCAL"
    assert any("local_rank ok=false" in entry for entry in logs)


def test_auto_statuses_runs_planned_stalled_when_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_powered_module(tmp_path)
    queue = tmp_path / "run_queue.csv"
    queue.write_text("run_name,config_path,status\nrun1,configs/a.yaml,planned\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/a.yaml").write_text("k: v\n", encoding="utf-8")

    commands: dict[str, str] = {}
    logs: list[str] = []

    monkeypatch.setattr(module, "_safe_api_key", lambda *_args, **_kwargs: "dummy")
    monkeypatch.setattr(module, "_safe_resolve_server_id_by_ip", lambda *_args, **_kwargs: "srv-1")
    monkeypatch.setattr(module, "ensure_server_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_safe_power_off", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_resolve_remote_repo", lambda *_args, **_kwargs: Path("/remote/repo"))
    monkeypatch.setattr(module, "_detect_remote_python", lambda *_args, **_kwargs: "python3")
    monkeypatch.setattr(module, "_sync_inputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_remote_rebuild_rollup", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_sync_rollup_back", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_remote_rank_and_sync", lambda *_args, **_kwargs: tmp_path / "rank.json")
    monkeypatch.setattr(module, "_run_with_retries", lambda fn, **_kwargs: fn())
    monkeypatch.setattr(
        module,
        "get_remote_queue_counts",
        lambda **_kwargs: {
            "ok": True,
            "queue_exists": True,
            "counts": {"planned": 3, "stalled": 1},
            "total": 4,
            "all_done": False,
            "has_metrics": False,
        },
    )

    def _remote_cmd(_host, _user, remote_command, *args, **kwargs):
        purpose = str(kwargs.get("command_purpose") or "")
        if purpose:
            commands[purpose] = str(remote_command)
        return 0

    monkeypatch.setattr(module, "_run_remote_command", _remote_cmd)
    monkeypatch.setattr(module, "_emit", lambda msg, _log_path, to_stderr=True: logs.append(msg))

    args = argparse.Namespace(
        queue=str(queue),
        compute_host="85.198.90.128",
        serverspace_server_id=None,
        ssh_user="root",
        ssh_port=22,
        preflight=False,
        remote_repo="auto",
        remote_repo_candidates=["/opt/coint4/coint4"],
        bootstrap_repo=False,
        bootstrap_remote_dir="/opt/coint4",
        bootstrap_venv=False,
        sync_inputs=False,
        sync_configs_bulk=False,
        force_remote_queue_overwrite=False,
        probe_queue=False,
        dry_run=False,
        statuses="auto",
        parallel=1,
        postprocess=True,
        max_retries=1,
        backoff_seconds=0.0,
        poweroff=True,
        wait_completion=False,
        wait_timeout_sec=300,
        wait_poll_sec=1,
    )

    rc = module.run_powered_queue(args, project_root=tmp_path, log_path=tmp_path / "powered.log")
    assert rc == 0
    queue_cmd = commands.get("queue-run", "")
    assert "--statuses planned,stalled" in queue_cmd
    assert any("auto_statuses" in entry and "chosen_statuses=planned,stalled" in entry for entry in logs)


def test_auto_statuses_watchdog_reclassifies_stale_running_and_starts_queue_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_powered_module(tmp_path)
    queue = tmp_path / "run_queue.csv"
    queue.write_text("run_name,config_path,status\nrun1,configs/a.yaml,planned\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/a.yaml").write_text("k: v\n", encoding="utf-8")

    commands: dict[str, str] = {}
    logs: list[str] = []
    probes = [
        {
            "ok": True,
            "queue_exists": True,
            "counts": {"running": 4},
            "total": 4,
            "all_done": False,
            "has_metrics": False,
        },
        {
            "ok": True,
            "queue_exists": True,
            "counts": {"running": 2, "stalled": 2},
            "total": 4,
            "all_done": False,
            "has_metrics": False,
        },
    ]
    watchdog_calls: list[dict] = []

    monkeypatch.setattr(module, "_safe_api_key", lambda *_args, **_kwargs: "dummy")
    monkeypatch.setattr(module, "_safe_resolve_server_id_by_ip", lambda *_args, **_kwargs: "srv-1")
    monkeypatch.setattr(module, "ensure_server_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_safe_power_off", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_resolve_remote_repo", lambda *_args, **_kwargs: Path("/remote/repo"))
    monkeypatch.setattr(module, "_detect_remote_python", lambda *_args, **_kwargs: "python3")
    monkeypatch.setattr(module, "_sync_inputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_remote_rebuild_rollup", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_sync_rollup_back", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_remote_rank_and_sync", lambda *_args, **_kwargs: tmp_path / "rank.json")
    monkeypatch.setattr(module, "_run_with_retries", lambda fn, **_kwargs: fn())
    monkeypatch.setattr(module, "_emit", lambda msg, _log_path, to_stderr=True: logs.append(msg))
    monkeypatch.setattr(module, "get_remote_queue_counts", lambda **_kwargs: probes.pop(0))
    monkeypatch.setattr(
        module,
        "_remote_watchdog_stale_running",
        lambda **kwargs: watchdog_calls.append(kwargs)
        or {
            "ok": True,
            "changed": 2,
            "stale_marked": 2,
            "running": 4,
            "rationale": "STALE_RUNNING_MARKED",
            "sample": [{"run_id": "runA", "age_sec": 1900}],
        },
    )

    def _remote_cmd(_host, _user, remote_command, *args, **kwargs):
        purpose = str(kwargs.get("command_purpose") or "")
        if purpose:
            commands[purpose] = str(remote_command)
        return 0

    monkeypatch.setattr(module, "_run_remote_command", _remote_cmd)

    args = argparse.Namespace(
        queue=str(queue),
        compute_host="85.198.90.128",
        serverspace_server_id=None,
        ssh_user="root",
        ssh_port=22,
        preflight=False,
        remote_repo="auto",
        remote_repo_candidates=["/opt/coint4/coint4"],
        bootstrap_repo=False,
        bootstrap_remote_dir="/opt/coint4",
        bootstrap_venv=False,
        sync_inputs=False,
        sync_configs_bulk=False,
        force_remote_queue_overwrite=False,
        probe_queue=False,
        dry_run=False,
        statuses="auto",
        parallel=1,
        postprocess=True,
        max_retries=1,
        backoff_seconds=0.0,
        poweroff=True,
        wait_completion=False,
        wait_timeout_sec=300,
        wait_poll_sec=1,
        watchdog=True,
        watchdog_stale_sec=900,
    )

    rc = module.run_powered_queue(args, project_root=tmp_path, log_path=tmp_path / "powered.log")
    assert rc == 0
    assert len(watchdog_calls) == 1
    queue_cmd = commands.get("queue-run", "")
    assert "--statuses planned,stalled" in queue_cmd
    assert any("watchdog checked" in entry and "changed=2" in entry for entry in logs)


def test_auto_statuses_skips_queue_run_when_only_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_powered_module(tmp_path)
    queue = tmp_path / "run_queue.csv"
    queue.write_text("run_name,config_path,status\nrun1,configs/a.yaml,planned\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/a.yaml").write_text("k: v\n", encoding="utf-8")

    events: list[str] = []
    monkeypatch.setattr(module, "_safe_api_key", lambda *_args, **_kwargs: "dummy")
    monkeypatch.setattr(module, "_safe_resolve_server_id_by_ip", lambda *_args, **_kwargs: "srv-1")
    monkeypatch.setattr(module, "ensure_server_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_safe_power_off", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_resolve_remote_repo", lambda *_args, **_kwargs: Path("/remote/repo"))
    monkeypatch.setattr(module, "_detect_remote_python", lambda *_args, **_kwargs: "python3")
    monkeypatch.setattr(module, "_sync_inputs", lambda *_args, **_kwargs: None)
    watchdog_calls: list[dict] = []
    monkeypatch.setattr(
        module,
        "get_remote_queue_counts",
        lambda **_kwargs: {
            "ok": True,
            "queue_exists": True,
            "counts": {"running": 4},
            "total": 4,
            "all_done": False,
            "has_metrics": False,
        },
    )
    monkeypatch.setattr(
        module,
        "_remote_watchdog_stale_running",
        lambda **kwargs: watchdog_calls.append(kwargs)
        or {
            "ok": True,
            "changed": 0,
            "stale_marked": 0,
            "running": 4,
            "rationale": "NO_STALE_RUNNING",
            "sample": [],
        },
    )
    monkeypatch.setattr(
        module,
        "_run_remote_command",
        lambda *_args, **_kwargs: events.append(str(_kwargs.get("command_purpose") or "")) or 0,
    )
    monkeypatch.setattr(module, "_remote_rebuild_rollup", lambda *_args, **_kwargs: events.append("remote_rollup"))
    monkeypatch.setattr(module, "_sync_rollup_back", lambda *_args, **_kwargs: events.append("sync_rollup_back"))
    monkeypatch.setattr(module, "_remote_rank_and_sync", lambda *_args, **_kwargs: events.append("remote_rank") or (tmp_path / "rank.json"))

    args = argparse.Namespace(
        queue=str(queue),
        compute_host="85.198.90.128",
        serverspace_server_id=None,
        ssh_user="root",
        ssh_port=22,
        preflight=False,
        remote_repo="auto",
        remote_repo_candidates=["/opt/coint4/coint4"],
        bootstrap_repo=False,
        bootstrap_remote_dir="/opt/coint4",
        bootstrap_venv=False,
        sync_inputs=False,
        sync_configs_bulk=False,
        force_remote_queue_overwrite=False,
        probe_queue=False,
        dry_run=False,
        statuses="auto",
        parallel=1,
        postprocess=True,
        max_retries=1,
        backoff_seconds=0.0,
        poweroff=True,
        wait_completion=False,
        wait_timeout_sec=300,
        wait_poll_sec=1,
        watchdog=True,
        watchdog_stale_sec=900,
    )

    rc = module.run_powered_queue(args, project_root=tmp_path, log_path=tmp_path / "powered.log")
    assert rc == 0
    assert len(watchdog_calls) == 1
    assert "queue-run" not in events
    assert "remote-postprocess-queue" in events
    assert "sync_rollup_back" in events
    assert "remote_rank" in events


def test_auto_statuses_completed_total_skips_queue_run_but_rollup_and_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_powered_module(tmp_path)
    queue = tmp_path / "run_queue.csv"
    queue.write_text("run_name,config_path,status\nrun1,configs/a.yaml,planned\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/a.yaml").write_text("k: v\n", encoding="utf-8")

    events: list[str] = []
    monkeypatch.setattr(module, "_safe_api_key", lambda *_args, **_kwargs: "dummy")
    monkeypatch.setattr(module, "_safe_resolve_server_id_by_ip", lambda *_args, **_kwargs: "srv-1")
    monkeypatch.setattr(module, "ensure_server_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_safe_power_off", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_resolve_remote_repo", lambda *_args, **_kwargs: Path("/remote/repo"))
    monkeypatch.setattr(module, "_detect_remote_python", lambda *_args, **_kwargs: "python3")
    monkeypatch.setattr(module, "_sync_inputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "get_remote_queue_counts",
        lambda **_kwargs: {
            "ok": True,
            "queue_exists": True,
            "counts": {"completed": 4},
            "total": 4,
            "all_done": True,
            "has_metrics": True,
        },
    )

    def _remote_cmd(*_args, **_kwargs):
        events.append(str(_kwargs.get("command_purpose") or ""))
        return 0

    monkeypatch.setattr(module, "_run_remote_command", _remote_cmd)
    monkeypatch.setattr(module, "_wait_for_completion", lambda *_args, **_kwargs: events.append("wait_completion"))
    monkeypatch.setattr(module, "_remote_rebuild_rollup", lambda *_args, **_kwargs: events.append("remote_rollup"))
    monkeypatch.setattr(module, "_sync_rollup_back", lambda *_args, **_kwargs: events.append("sync_rollup_back"))
    monkeypatch.setattr(module, "_remote_rank_and_sync", lambda *_args, **_kwargs: events.append("remote_rank") or (tmp_path / "rank.json"))

    args = argparse.Namespace(
        queue=str(queue),
        compute_host="85.198.90.128",
        serverspace_server_id=None,
        ssh_user="root",
        ssh_port=22,
        preflight=False,
        remote_repo="auto",
        remote_repo_candidates=["/opt/coint4/coint4"],
        bootstrap_repo=False,
        bootstrap_remote_dir="/opt/coint4",
        bootstrap_venv=False,
        sync_inputs=False,
        sync_configs_bulk=False,
        force_remote_queue_overwrite=False,
        probe_queue=False,
        dry_run=False,
        statuses="auto",
        parallel=1,
        postprocess=True,
        max_retries=1,
        backoff_seconds=0.0,
        poweroff=True,
        wait_completion=True,
        wait_timeout_sec=300,
        wait_poll_sec=1,
    )

    rc = module.run_powered_queue(args, project_root=tmp_path, log_path=tmp_path / "powered.log")
    assert rc == 0
    assert "queue-run" not in events
    assert "wait_completion" not in events
    assert "remote-postprocess-queue" in events
    assert "sync_rollup_back" in events
    assert "remote_rank" in events
