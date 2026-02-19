from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import csv
from pathlib import Path
from types import SimpleNamespace


def _load_autonomous_module(tmp_path: Path):
    script_path = Path(__file__).resolve().parents[2] / "scripts/optimization/autonomous_optimize.py"
    spec = importlib.util.spec_from_file_location(f"autonomous_optimize_test_{tmp_path.name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        once=False,
        until_done=False,
        max_iterations=None,
        use_codex_exec=False,
        codex_model="",
        plan_only=False,
        wait_timeout_sec=21600,
        wait_poll_sec=60,
        local_rollup_rebuild=False,
    )


def _setup_runner(module, tmp_path: Path):
    repo_root = tmp_path / "repo"
    app_root = repo_root / "coint4"

    (app_root / "configs" / "autopilot").mkdir(parents=True, exist_ok=True)
    (app_root / "configs" / "autopilot" / "budget1000_batch_loop_bridge11_20260217.yaml").write_text(
        "search:\n  max_rounds: 5\nselection:\n  min_windows: 1\n  min_trades: 1\n  min_pairs: 1\n  max_dd_pct: 0.99\n",
        encoding="utf-8",
    )
    (repo_root / "docs").mkdir(parents=True, exist_ok=True)
    (app_root / "artifacts" / "optimization_state").mkdir(parents=True, exist_ok=True)
    (app_root / "artifacts" / "wfa" / "aggregate" / "rollup").mkdir(parents=True, exist_ok=True)

    schema_src = Path(__file__).resolve().parents[2] / "scripts/optimization/schemas/autopilot_decision.schema.json"
    schema_dst = app_root / "scripts" / "optimization" / "schemas" / "autopilot_decision.schema.json"
    schema_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(schema_src, schema_dst)

    runner = module.AutonomousOptimizer(_args())
    runner.repo_root = repo_root
    runner.app_root = app_root
    runner.state_dir = app_root / "artifacts" / "optimization_state"
    runner.state_path = runner.state_dir / "autonomous_state.json"
    runner.main_log_path = runner.state_dir / "autonomous_service.log"
    runner.decisions_dir = runner.state_dir / "decisions"
    runner.rollup_csv = app_root / "artifacts" / "wfa" / "aggregate" / "rollup" / "run_index.csv"
    runner.bridge11_path = app_root / "configs" / "autopilot" / "budget1000_batch_loop_bridge11_20260217.yaml"
    runner.baseline_queue = app_root / "artifacts" / "wfa" / "aggregate" / "baseline" / "run_queue.csv"
    runner.best_params_path = repo_root / "docs" / "best_params_latest.yaml"
    runner.final_report_path = repo_root / "docs" / "final_report_latest.md"
    runner.rank_results_dir = runner.state_dir / "rank_results"
    runner.python_exec = Path(sys.executable)
    runner.powered_runner = app_root / "scripts" / "optimization" / "run_wfa_queue_powered.py"
    runner.ensure_next_batch = app_root / "scripts" / "optimization" / "loop_orchestrator" / "ensure_next_batch.py"
    runner.build_run_index = app_root / "scripts" / "optimization" / "build_run_index.py"
    runner.rank_script = app_root / "scripts" / "optimization" / "rank_multiwindow_robust_runs.py"
    runner.codex_schema_path = schema_dst

    runner.state_dir.mkdir(parents=True, exist_ok=True)
    runner.decisions_dir.mkdir(parents=True, exist_ok=True)
    runner.rank_results_dir.mkdir(parents=True, exist_ok=True)
    runner.best_params_path.parent.mkdir(parents=True, exist_ok=True)
    runner.final_report_path.parent.mkdir(parents=True, exist_ok=True)
    runner.stop_policy = runner._load_stop_policy()
    return runner


def _decision_payload(
    *,
    decision_id: str,
    next_action: str,
    stop: bool = False,
    stop_reason: str = "",
    human_explanation_md: str = "explain",
    next_run_group: str = "rg",
    next_queue_path: str = "coint4/artifacts/wfa/aggregate/rg/run_queue.csv",
    queue_entries: list[dict] | None = None,
) -> dict:
    if queue_entries is None:
        queue_entries = []
    return {
        "decision_id": decision_id,
        "stop": stop,
        "stop_reason": stop_reason,
        "human_explanation_md": human_explanation_md,
        "next_run_group": next_run_group,
        "next_queue_path": next_queue_path,
        "queue_entries": queue_entries,
        "file_edits": [],
        "constraints": {
            "must_keep": ["do_not_change_metrics_formula"],
            "allow_anything_in_repo": True,
        },
        "next_action": next_action,
        "wait_seconds": 60,
    }


def _mock_codex_subprocess(decisions: list[dict]):
    calls = {"n": 0}

    def _run(cmd, *args, **kwargs):
        assert cmd[0] == "codex"
        assert "--output-last-message" in cmd
        idx = calls["n"]
        calls["n"] += 1
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        payload = decisions[min(idx, len(decisions) - 1)]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout='{"ok":true}\n', stderr="")

    return _run


def _write_rollup(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "run_group",
        "results_dir",
        "metrics_path",
        "config_path",
        "status",
        "metrics_present",
        "sharpe_ratio_abs",
        "max_drawdown_on_equity",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_valid_codex_decision_creates_queue_and_configs(tmp_path: Path, monkeypatch) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)

    queue_entries = [
        {
            "config_path": "coint4/configs/autopilot/generated/test_batch/run1.yaml",
            "status": "planned",
            "results_dir": "artifacts/wfa/runs_clean/rg_codex/run1",
            "notes": "baseline candidate",
            "overrides": {"portfolio": {"risk_per_position_pct": 0.006}},
        }
    ]
    decisions = [
        _decision_payload(
            decision_id="d1",
            next_action="run_next_batch",
            next_run_group="rg_codex",
            next_queue_path="coint4/artifacts/wfa/aggregate/rg_codex/run_queue.csv",
            queue_entries=queue_entries,
            human_explanation_md="### Run batch\n- candidate set",
        ),
        _decision_payload(
            decision_id="d2",
            next_action="wait",
            stop=False,
            stop_reason="need metrics",
            human_explanation_md="### Wait\n- awaiting completion",
            next_run_group="rg_codex",
            next_queue_path="coint4/artifacts/wfa/aggregate/rg_codex/run_queue.csv",
            queue_entries=[],
        ),
    ]
    monkeypatch.setattr(module.subprocess, "run", _mock_codex_subprocess(decisions))
    monkeypatch.setattr(runner, "_run_powered_queue", lambda queue, iteration_log: 0)
    monkeypatch.setattr(runner, "_quarantine_demo_queues", lambda: None)

    state = runner._default_state()
    result = runner.run_iteration(state=state)

    queue_path = runner.repo_root / "coint4/artifacts/wfa/aggregate/rg_codex/run_queue.csv"
    config_path = runner.repo_root / "coint4/configs/autopilot/generated/test_batch/run1.yaml"
    assert queue_path.exists()
    assert config_path.exists()
    assert result["iteration"] == 1
    assert result["current_run_group"] == "rg_codex"
    assert result["last_iteration_phase"] == "waiting_codex"


def test_codex_stop_marks_state_done(tmp_path: Path, monkeypatch) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)
    decisions = [
        _decision_payload(
            decision_id="stop-1",
            next_action="stop",
            stop=True,
            stop_reason="no further improvement",
            human_explanation_md="### Stop\n- plateau reached",
            queue_entries=[],
        )
    ]
    monkeypatch.setattr(module.subprocess, "run", _mock_codex_subprocess(decisions))
    monkeypatch.setattr(runner, "_quarantine_demo_queues", lambda: None)
    monkeypatch.setattr(runner, "_run_powered_queue", lambda queue, iteration_log: (_ for _ in ()).throw(AssertionError("powered must not run")))

    rc = runner.run()
    assert rc == 0
    state = runner.load_state()
    assert state["status"] == "done"
    assert state["last_decision_id"] == "stop-1"


def test_codex_wait_logs_human_explanation_and_memo(tmp_path: Path, monkeypatch) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)
    explanation = "### Waiting\n- insufficient completed runs"
    decisions = [
        _decision_payload(
            decision_id="wait-1",
            next_action="wait",
            stop=False,
            stop_reason="need more completed runs",
            human_explanation_md=explanation,
            queue_entries=[],
        )
    ]
    monkeypatch.setattr(module.subprocess, "run", _mock_codex_subprocess(decisions))
    monkeypatch.setattr(runner, "_quarantine_demo_queues", lambda: None)

    rc = runner.run()
    assert rc == 0
    state = runner.load_state()
    assert state["last_error"] == "need more completed runs"
    assert state["last_iteration_phase"] == "waiting_codex"

    log_text = runner.main_log_path.read_text(encoding="utf-8")
    assert "codex decision_id=wait-1 next_action=wait stop=False" in log_text
    assert "### Waiting" in log_text
    assert "- insufficient completed runs" in log_text

    memos = sorted(runner.decisions_dir.glob("decision_*.md"))
    assert memos
    memo_text = memos[-1].read_text(encoding="utf-8")
    assert explanation in memo_text


def test_invalid_codex_json_goes_wait_without_queue(tmp_path: Path, monkeypatch) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)

    def _broken_run(cmd, *args, **kwargs):
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("{invalid json\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _broken_run)
    monkeypatch.setattr(runner, "_quarantine_demo_queues", lambda: None)

    rc = runner.run()
    assert rc == 0
    state = runner.load_state()
    assert state["last_iteration_phase"] == "waiting_codex"
    assert str(state["last_error"]).startswith("CODEX_DECISION_INVALID_JSON")

    queues = list((runner.app_root / "artifacts" / "wfa" / "aggregate").glob("*/run_queue*.csv"))
    assert not queues


def test_build_decision_context_has_completed_evidence(tmp_path: Path) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)

    run_group = "rg_ctx_ready"
    queue_path = runner.app_root / "artifacts" / "wfa" / "aggregate" / run_group / "run_queue.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        "\n".join(
            [
                "config_path,results_dir,status",
                "configs/autopilot/generated/rg_ctx_ready/run1.yaml,artifacts/wfa/runs_clean/rg_ctx_ready/run1,planned",
                "configs/autopilot/generated/rg_ctx_ready/run2.yaml,artifacts/wfa/runs_clean/rg_ctx_ready/run2,planned",
                "configs/autopilot/generated/rg_ctx_ready/run3.yaml,artifacts/wfa/runs_clean/rg_ctx_ready/run3,planned",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _write_rollup(
        runner.rollup_csv,
        [
            {
                "run_id": "run1",
                "run_group": run_group,
                "results_dir": "artifacts/wfa/runs_clean/rg_ctx_ready/run1",
                "metrics_path": "a.csv",
                "config_path": "configs/autopilot/generated/rg_ctx_ready/run1.yaml",
                "status": "completed",
                "metrics_present": "True",
                "sharpe_ratio_abs": "0.70",
                "max_drawdown_on_equity": "-0.10",
            },
            {
                "run_id": "run2",
                "run_group": run_group,
                "results_dir": "artifacts/wfa/runs_clean/rg_ctx_ready/run2",
                "metrics_path": "b.csv",
                "config_path": "configs/autopilot/generated/rg_ctx_ready/run2.yaml",
                "status": "completed",
                "metrics_present": "True",
                "sharpe_ratio_abs": "0.95",
                "max_drawdown_on_equity": "-0.12",
            },
            {
                "run_id": "run3",
                "run_group": run_group,
                "results_dir": "artifacts/wfa/runs_clean/rg_ctx_ready/run3",
                "metrics_path": "c.csv",
                "config_path": "configs/autopilot/generated/rg_ctx_ready/run3.yaml",
                "status": "completed",
                "metrics_present": "True",
                "sharpe_ratio_abs": "1.25",
                "max_drawdown_on_equity": "-0.08",
            },
        ],
    )

    state = runner._default_state()
    state["current_run_group"] = run_group
    context = runner._build_decision_context(state)
    evidence = context["evidence"]

    assert evidence["completed_count"] == 3
    assert evidence["metrics_present_true_completed"] == 3
    assert evidence["decision_data_ready"] is True
    assert "NO_COMPLETED_RUNS" not in set(evidence["computed_missing_reasons"])
    assert evidence["top_candidates"][0]["run_id"] == "run3"


def test_build_decision_context_marks_no_completed_runs(tmp_path: Path) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)

    run_group = "rg_ctx_wait"
    queue_path = runner.app_root / "artifacts" / "wfa" / "aggregate" / run_group / "run_queue.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        "\n".join(
            [
                "config_path,results_dir,status",
                "configs/autopilot/generated/rg_ctx_wait/run1.yaml,artifacts/wfa/runs_clean/rg_ctx_wait/run1,planned",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _write_rollup(
        runner.rollup_csv,
        [
            {
                "run_id": "run1",
                "run_group": run_group,
                "results_dir": "artifacts/wfa/runs_clean/rg_ctx_wait/run1",
                "metrics_path": "a.csv",
                "config_path": "configs/autopilot/generated/rg_ctx_wait/run1.yaml",
                "status": "planned",
                "metrics_present": "False",
                "sharpe_ratio_abs": "",
                "max_drawdown_on_equity": "",
            }
        ],
    )

    state = runner._default_state()
    state["current_run_group"] = run_group
    context = runner._build_decision_context(state)
    evidence = context["evidence"]

    assert evidence["completed_count"] == 0
    assert "NO_COMPLETED_RUNS" in set(evidence["computed_missing_reasons"])
    assert "NO_COMPLETED_RUNS" in set(evidence["blocking_missing_reasons"])


def test_build_decision_context_marks_rollup_lagging_remote(tmp_path: Path, monkeypatch) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)

    run_group = "rg_ctx_remote_lag"
    queue_path = runner.app_root / "artifacts" / "wfa" / "aggregate" / run_group / "run_queue.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        "\n".join(
            [
                "config_path,results_dir,status",
                "configs/autopilot/generated/rg_ctx_remote_lag/run1.yaml,artifacts/wfa/runs_clean/rg_ctx_remote_lag/run1,planned",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_rollup(
        runner.rollup_csv,
        [
            {
                "run_id": "run1",
                "run_group": run_group,
                "results_dir": "artifacts/wfa/runs_clean/rg_ctx_remote_lag/run1",
                "metrics_path": "a.csv",
                "config_path": "configs/autopilot/generated/rg_ctx_remote_lag/run1.yaml",
                "status": "planned",
                "metrics_present": "False",
                "sharpe_ratio_abs": "",
                "max_drawdown_on_equity": "",
            }
        ],
    )

    monkeypatch.setattr(runner, "_is_powered_runner_active_for_queue", lambda _q: True)
    monkeypatch.setattr(
        runner,
        "_probe_remote_queue_counts",
        lambda _q: {"counts": {"completed": 2, "running": 3}, "total": 5, "has_metrics": True},
    )

    state = runner._default_state()
    state["current_run_group"] = run_group
    context = runner._build_decision_context(state)
    evidence = context["evidence"]

    assert evidence["completed_count"] == 0
    assert "NO_COMPLETED_RUNS" not in set(evidence["blocking_missing_reasons"])
    assert "ROLLUP_LAGGING_REMOTE" in set(evidence["blocking_missing_reasons"])
    assert evidence["remote_counts"]["completed"] == 2


def test_log_progress_snapshot_rate_limited(tmp_path: Path, monkeypatch) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)

    run_group = "rg_progress"
    queue_path = runner.app_root / "artifacts" / "wfa" / "aggregate" / run_group / "run_queue.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        "\n".join(
            [
                "config_path,results_dir,status",
                "configs/autopilot/generated/rg_progress/run1.yaml,artifacts/wfa/runs_clean/rg_progress/run1,planned",
                "configs/autopilot/generated/rg_progress/run2.yaml,artifacts/wfa/runs_clean/rg_progress/run2,stalled",
                "configs/autopilot/generated/rg_progress/run3.yaml,artifacts/wfa/runs_clean/rg_progress/run3,planned",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_rollup(
        runner.rollup_csv,
        [
            {
                "run_id": "run1",
                "run_group": run_group,
                "results_dir": "artifacts/wfa/runs_clean/rg_progress/run1",
                "metrics_path": "a.csv",
                "config_path": "configs/autopilot/generated/rg_progress/run1.yaml",
                "status": "planned",
                "metrics_present": "False",
                "sharpe_ratio_abs": "",
                "max_drawdown_on_equity": "",
            },
            {
                "run_id": "run2",
                "run_group": run_group,
                "results_dir": "artifacts/wfa/runs_clean/rg_progress/run2",
                "metrics_path": "b.csv",
                "config_path": "configs/autopilot/generated/rg_progress/run2.yaml",
                "status": "running",
                "metrics_present": "False",
                "sharpe_ratio_abs": "",
                "max_drawdown_on_equity": "",
            },
            {
                "run_id": "run3",
                "run_group": run_group,
                "results_dir": "artifacts/wfa/runs_clean/rg_progress/run3",
                "metrics_path": "c.csv",
                "config_path": "configs/autopilot/generated/rg_progress/run3.yaml",
                "status": "completed",
                "metrics_present": "True",
                "sharpe_ratio_abs": "1.2",
                "max_drawdown_on_equity": "-0.1",
            },
        ],
    )

    state = runner._default_state()
    state["current_run_group"] = run_group
    state["iteration_started_utc"] = "2026-02-19T12:00:00Z"

    monkeypatch.setattr(module, "_utc_now", lambda: "2026-02-19T12:00:00Z")
    wrote_first = runner.log_progress_snapshot(state, phase="wait")
    wrote_second = runner.log_progress_snapshot(state, phase="wait")
    assert wrote_first is True
    assert wrote_second is False

    monkeypatch.setattr(module, "_utc_now", lambda: "2026-02-19T12:01:05Z")
    wrote_third = runner.log_progress_snapshot(state, phase="wait")
    assert wrote_third is True

    text = runner.main_log_path.read_text(encoding="utf-8")
    assert text.count("progress phase=wait") == 2


def test_log_progress_snapshot_includes_remote_counts(tmp_path: Path, monkeypatch) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)

    run_group = "rg_remote_progress"
    queue_path = runner.app_root / "artifacts" / "wfa" / "aggregate" / run_group / "run_queue.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        "\n".join(
            [
                "config_path,results_dir,status",
                "configs/autopilot/generated/rg_remote_progress/run1.yaml,artifacts/wfa/runs_clean/rg_remote_progress/run1,planned",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_rollup(
        runner.rollup_csv,
        [
            {
                "run_id": "run1",
                "run_group": run_group,
                "results_dir": "artifacts/wfa/runs_clean/rg_remote_progress/run1",
                "metrics_path": "a.csv",
                "config_path": "configs/autopilot/generated/rg_remote_progress/run1.yaml",
                "status": "planned",
                "metrics_present": "False",
                "sharpe_ratio_abs": "",
                "max_drawdown_on_equity": "",
            }
        ],
    )

    state = runner._default_state()
    state["current_run_group"] = run_group
    state["last_iteration_phase"] = "waiting_codex"
    state["iteration_started_utc"] = "2026-02-19T12:00:00Z"

    monkeypatch.setattr(module, "_utc_now", lambda: "2026-02-19T12:01:00Z")
    monkeypatch.setattr(runner, "_is_powered_runner_active_for_queue", lambda _q: True)
    monkeypatch.setattr(
        runner,
        "_probe_remote_queue_counts",
        lambda _q: {
            "counts": {"running": 4, "completed": 2},
            "total": 6,
            "has_metrics": True,
        },
    )

    wrote = runner.log_progress_snapshot(state, phase="wait:waiting_codex", min_interval_sec=1)
    assert wrote is True
    text = runner.main_log_path.read_text(encoding="utf-8")
    assert "remote_counts=" in text
    assert "remote_total=6" in text
    assert "remote_has_metrics=true" in text


def test_wait_no_completed_launches_data_collection(tmp_path: Path, monkeypatch) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)

    run_group = "rg_wait_launch"
    queue_path = runner.app_root / "artifacts" / "wfa" / "aggregate" / run_group / "run_queue.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        "\n".join(
            [
                "config_path,results_dir,status",
                "configs/autopilot/generated/rg_wait_launch/run1.yaml,artifacts/wfa/runs_clean/rg_wait_launch/run1,planned",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_rollup(
        runner.rollup_csv,
        [
            {
                "run_id": "run1",
                "run_group": run_group,
                "results_dir": "artifacts/wfa/runs_clean/rg_wait_launch/run1",
                "metrics_path": "a.csv",
                "config_path": "configs/autopilot/generated/rg_wait_launch/run1.yaml",
                "status": "planned",
                "metrics_present": "False",
                "sharpe_ratio_abs": "",
                "max_drawdown_on_equity": "",
            }
        ],
    )

    decision = _decision_payload(
        decision_id="wait-launch",
        next_action="wait",
        stop=False,
        stop_reason="NO_COMPLETED_RUNS",
        human_explanation_md="wait",
        next_run_group=run_group,
        next_queue_path=f"coint4/artifacts/wfa/aggregate/{run_group}/run_queue.csv",
        queue_entries=[],
    )
    monkeypatch.setattr(module.subprocess, "run", _mock_codex_subprocess([decision]))
    monkeypatch.setattr(runner, "_quarantine_demo_queues", lambda: None)
    monkeypatch.setattr(runner, "_is_powered_runner_active_for_queue", lambda _q: False)

    launched = {"count": 0}

    def _fake_powered(queue, iteration_log):
        launched["count"] += 1
        assert queue.run_group == run_group
        assert queue.queue_path == queue_path
        assert iteration_log.exists() or iteration_log.parent.exists()
        return 0

    monkeypatch.setattr(runner, "_run_powered_queue", _fake_powered)

    state = runner._default_state()
    state["current_run_group"] = run_group
    result = runner.run_iteration(state=state)

    assert launched["count"] == 1
    assert result["last_iteration_phase"] == "waiting_codex"
    log_text = runner.main_log_path.read_text(encoding="utf-8")
    assert "autonomous: launched data-collection run for queue because NO_COMPLETED_RUNS" in log_text


def test_run_powered_queue_uses_compute_parallel_env(tmp_path: Path, monkeypatch) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)
    queue = runner.app_root / "artifacts" / "wfa" / "aggregate" / "rg_parallel_env" / "run_queue.csv"
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text(
        "run_name,config_path,status\nrun1,configs/autopilot/generated/rg_parallel_env/run1.yaml,planned\n",
        encoding="utf-8",
    )
    target = module.QueueTarget(
        run_group="rg_parallel_env",
        queue_path=queue,
        source="test",
        ready_rows=1,
    )
    iteration_log = runner._iteration_log_path(target.run_group, 0)

    captured: dict[str, list[str]] = {}

    def _fake_subprocess(*, cmd, cwd, iteration_log, env=None):
        captured["cmd"] = list(cmd)
        class _Proc:
            returncode = 0
            stdout = ""
            stderr = ""
        return _Proc()

    monkeypatch.setattr(runner, "_run_subprocess", _fake_subprocess)
    monkeypatch.setenv("COINT4_COMPUTE_PARALLEL", "auto")
    monkeypatch.setenv("COINT4_WATCHDOG", "true")
    monkeypatch.setenv("COINT4_WATCHDOG_STALE_SEC", "900")

    rc = runner._run_powered_queue(target, iteration_log)
    assert rc == 0
    cmd = captured["cmd"]
    assert "--statuses" in cmd
    assert cmd[cmd.index("--statuses") + 1] == "auto"
    assert "--parallel" in cmd
    assert cmd[cmd.index("--parallel") + 1] == "auto"
    assert "--watchdog" in cmd
    assert cmd[cmd.index("--watchdog") + 1] == "true"
    assert "--watchdog-stale-sec" in cmd
    assert cmd[cmd.index("--watchdog-stale-sec") + 1] == "900"
