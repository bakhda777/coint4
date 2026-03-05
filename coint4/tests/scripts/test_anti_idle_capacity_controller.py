from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module(script_name: str, tmp_path: Path):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_{tmp_path.name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_queue(path: Path, *, config_path: str, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config_path", "status"])
        writer.writerow([config_path, status])


def test_anti_idle_policy_and_process_slo_kpi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "app"
    aggregate_root = root / "artifacts" / "wfa" / "aggregate"
    state_dir = aggregate_root / ".autonomous"
    queue_path = aggregate_root / "group_a" / "run_queue.csv"

    config_rel = "configs/sample.yaml"
    config_abs = root / config_rel
    config_abs.parent.mkdir(parents=True, exist_ok=True)
    config_abs.write_text("name: sample\n", encoding="utf-8")
    _write_queue(queue_path, config_path=config_rel, status="planned")

    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "capacity_controller_state.json").write_text(
        json.dumps({"remote": {"reachable": True, "runner_count": 0}}, ensure_ascii=False),
        encoding="utf-8",
    )

    process_module = _load_module("process_slo_guard_agent.py", tmp_path)
    monkeypatch.setattr(process_module, "detect_local_runner_count", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["process_slo_guard_agent.py", "--root", str(root)])
    assert process_module.main() == 0

    process_state = json.loads((state_dir / "process_slo_state.json").read_text(encoding="utf-8"))
    assert process_state["kpi"]["executable_pending_rows"] == 1
    assert process_state["kpi"]["local_runner_count"] == 0
    assert process_state["kpi"]["remote_runner_count"] == 0
    assert process_state["kpi"]["idle_with_executable_pending"] is True

    capacity_module = _load_module("vps_capacity_controller_agent.py", tmp_path)
    monkeypatch.setattr(capacity_module, "detect_remote_load", lambda *_args, **_kwargs: 3.0)
    monkeypatch.setattr(capacity_module, "detect_remote_runner_count", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(sys, "argv", ["vps_capacity_controller_agent.py", "--root", str(root)])
    assert capacity_module.main() == 0

    capacity_state = json.loads((state_dir / "capacity_controller_state.json").read_text(encoding="utf-8"))
    assert capacity_state["backlog"]["executable_pending"] == 1
    assert "anti_idle_executable_backlog" in capacity_state["reasons"]
    assert int(capacity_state["policy"]["search_parallel_min"]) >= 16
    assert int(capacity_state["policy"]["search_parallel_max"]) >= 48
