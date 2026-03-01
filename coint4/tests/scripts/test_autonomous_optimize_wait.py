from __future__ import annotations

import importlib.util
import json
import shutil
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
        planner_mode="legacy",
        use_codex_exec=True,
        codex_model="",
        evolution_controller_group="",
        evolution_run_prefix="autonomous_evo",
        evolution_contains=[],
        evolution_num_variants=12,
        evolution_ir_mode="patch_ast",
        evolution_policy_scale="auto",
        evolution_ast_max_complexity_score=60.0,
        evolution_ast_max_redundancy_similarity=0.85,
        evolution_patch_max_attempts=8,
        evolution_llm_model="gpt-5.2",
        evolution_llm_effort="xhigh",
        evolution_llm_timeout_sec=180,
        evolution_llm_verify_semantic=True,
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
    runner.evolve_next_batch = app_root / "scripts" / "optimization" / "evolve_next_batch.py"
    runner.reflect_next_action = app_root / "scripts" / "optimization" / "reflect_next_action.py"
    runner.build_factor_pool = app_root / "scripts" / "optimization" / "build_factor_pool.py"
    runner.codex_schema_path = schema_dst
    runner._ensure_codex_auth_ready = lambda: True
    runner._run_codex_json = lambda *args, **kwargs: None

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


def _mock_decide_with_codex(module, runner, payloads: list[dict | None]):
    calls = {"n": 0}

    def _decide(_state):
        idx = calls["n"]
        calls["n"] += 1
        payload = payloads[min(idx, len(payloads) - 1)] if payloads else None
        if payload is None:
            return None

        stamp = f"test_{idx + 1:03d}"
        decision_json_path = runner.decisions_dir / f"decision_{stamp}.json"
        context_path = runner.decisions_dir / f"context_{stamp}.json"
        decision_md_path = runner.decisions_dir / f"decision_{stamp}.md"
        exec_log_path = runner.decisions_dir / f"codex_exec_{stamp}.jsonl"
        decision_json_path.parent.mkdir(parents=True, exist_ok=True)
        decision_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        context_path.write_text("{}", encoding="utf-8")
        exec_log_path.write_text("", encoding="utf-8")

        runner._persist_decision_memo(decision_payload=payload, decision_md_path=decision_md_path)
        runner.log(
            "codex decision_id={decision_id} next_action={next_action} stop={stop}".format(
                decision_id=str(payload.get("decision_id") or "").strip(),
                next_action=str(payload.get("next_action") or "").strip(),
                stop=bool(payload.get("stop")),
            )
        )
        runner._log_human_explanation(str(payload.get("human_explanation_md") or ""))
        return module.CodexDecision(
            payload=payload,
            decision_json_path=decision_json_path,
            context_path=context_path,
            decision_md_path=decision_md_path,
            exec_log_path=exec_log_path,
        )

    return _decide


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
        )
    ]
    monkeypatch.setattr(runner, "decide_with_codex", _mock_decide_with_codex(module, runner, decisions))
    monkeypatch.setattr(runner, "_run_powered_queue", lambda queue, iteration_log: 0)
    monkeypatch.setattr(
        runner,
        "_rank_from_remote_result",
        lambda _rg: module.RankResult(
            ok=True,
            source="test",
            score=1.23,
            worst_robust_sharpe=1.23,
            worst_dd_pct=-0.08,
            run_name="run1",
            config_path="configs/autopilot/generated/rg_codex/run1.yaml",
            details="ok",
        ),
    )
    monkeypatch.setattr(runner, "_quarantine_demo_queues", lambda: None)

    state = runner._default_state()
    result = runner.run_iteration(state=state)

    queue_path = runner.repo_root / "coint4/artifacts/wfa/aggregate/rg_codex/run_queue.csv"
    assert queue_path.exists()
    with queue_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    generated_config_rel = str(rows[0].get("config_path") or "").strip()
    assert generated_config_rel
    config_path = runner.app_root / generated_config_rel
    assert config_path.exists()
    assert result["iteration"] == 1
    assert result["current_run_group"] == "rg_codex"
    assert result["last_iteration_phase"] == "rank_ok"


def test_codex_run_iteration_uses_single_decision_per_cycle(tmp_path: Path, monkeypatch) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)

    queue_entries = [
        {
            "config_path": "coint4/configs/autopilot/generated/rg_one_decision/run1.yaml",
            "status": "planned",
            "results_dir": "artifacts/wfa/runs_clean/rg_one_decision/run1",
            "notes": "single decision cycle",
            "overrides": {"portfolio": {"risk_per_position_pct": 0.006}},
        }
    ]
    decision = _decision_payload(
        decision_id="single-1",
        next_action="run_next_batch",
        next_run_group="rg_one_decision",
        next_queue_path="coint4/artifacts/wfa/aggregate/rg_one_decision/run_queue.csv",
        queue_entries=queue_entries,
        human_explanation_md="run one batch",
    )

    calls = {"n": 0}

    decide = _mock_decide_with_codex(module, runner, [decision])

    def _counting_decide(state):
        calls["n"] += 1
        return decide(state)

    monkeypatch.setattr(runner, "decide_with_codex", _counting_decide)
    monkeypatch.setattr(runner, "_run_powered_queue", lambda queue, iteration_log: 0)
    monkeypatch.setattr(
        runner,
        "_rank_from_remote_result",
        lambda _rg: module.RankResult(
            ok=True,
            source="test",
            score=1.11,
            worst_robust_sharpe=1.11,
            worst_dd_pct=-0.05,
            run_name="run1",
            config_path="configs/autopilot/generated/rg_one_decision/run1.yaml",
            details="ok",
        ),
    )
    monkeypatch.setattr(runner, "_quarantine_demo_queues", lambda: None)

    state = runner._default_state()
    result = runner.run_iteration(state=state)

    assert calls["n"] == 1
    assert result["last_iteration_phase"] == "rank_ok"


def test_evolution_mode_dispatches_to_evolution_iteration(tmp_path: Path, monkeypatch) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)
    runner.args.planner_mode = "evolution"

    called = {"n": 0}

    def _fake_evolution(*, state):
        called["n"] += 1
        state["last_iteration_phase"] = "rank_ok"
        return state

    monkeypatch.setattr(runner, "_run_iteration_evolution", _fake_evolution)
    monkeypatch.setattr(runner, "_run_iteration_local", lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy path should not run")))
    monkeypatch.setattr(runner, "decide_with_codex", lambda _state: (_ for _ in ()).throw(AssertionError("codex path should not run")))

    state = runner._default_state()
    result = runner.run_iteration(state=state)

    assert called["n"] == 1
    assert result["last_iteration_phase"] == "rank_ok"


def test_plan_next_queue_with_evolution_reads_decision_queue(tmp_path: Path, monkeypatch) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)
    runner.args.planner_mode = "evolution"
    runner.args.use_codex_exec = False
    runner.args.evolution_contains = []
    runner.args.evolution_controller_group = ""

    runner.evolve_next_batch.parent.mkdir(parents=True, exist_ok=True)
    runner.evolve_next_batch.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    decision_run_group = "rg_evo_test"
    decision_queue_rel = f"artifacts/wfa/aggregate/{decision_run_group}/run_queue.csv"
    decision_queue_path = runner.app_root / decision_queue_rel
    decision_queue_path.parent.mkdir(parents=True, exist_ok=True)
    decision_queue_path.write_text(
        "config_path,results_dir,status\nconfigs/a.yaml,artifacts/wfa/runs_clean/rg_evo_test/run1,planned\n",
        encoding="utf-8",
    )

    def _fake_run_subprocess(*, cmd, cwd, iteration_log, env=None):
        assert "evolve_next_batch.py" in " ".join(cmd)
        idx = cmd.index("--controller-group")
        controller_group = cmd[idx + 1]
        decision_dir = runner.app_root / "artifacts" / "wfa" / "aggregate" / controller_group / "decisions"
        decision_dir.mkdir(parents=True, exist_ok=True)
        (decision_dir / "decision_test.json").write_text(
            json.dumps(
                {
                    "decision_id": "evo-d1",
                    "run_group": decision_run_group,
                    "queue_path": decision_queue_rel,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        class _Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Proc()

    monkeypatch.setattr(runner, "_run_subprocess", _fake_run_subprocess)

    state = runner._default_state()
    iteration_log = runner._iteration_log_path("rg_plan_test", 0)
    plan = runner._plan_next_queue_with_evolution(
        state=state,
        iteration=0,
        run_group=decision_run_group,
        iteration_log=iteration_log,
    )

    assert plan.queue.run_group == decision_run_group
    assert plan.queue.queue_path == decision_queue_path
    assert plan.queue.ready_rows == 1
    assert state["last_decision_id"] == "evo-d1"
    assert state["last_decision_action"] == "run_next_batch_evolution"


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
    monkeypatch.setattr(runner, "decide_with_codex", _mock_decide_with_codex(module, runner, decisions))
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
    monkeypatch.setattr(runner, "decide_with_codex", _mock_decide_with_codex(module, runner, decisions))
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


def test_codex_wait_records_trajectory_memory(tmp_path: Path, monkeypatch) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)
    decisions = [
        _decision_payload(
            decision_id="wait-memory-1",
            next_action="wait",
            stop=False,
            stop_reason="need more completed runs",
            human_explanation_md="",
            queue_entries=[],
        )
    ]
    monkeypatch.setattr(runner, "decide_with_codex", _mock_decide_with_codex(module, runner, decisions))
    monkeypatch.setattr(runner, "_quarantine_demo_queues", lambda: None)

    state = runner._default_state()
    result = runner.run_iteration(state=state)
    memory = result.get("trajectory_memory")

    assert isinstance(memory, list) and memory
    entry = memory[-1]
    assert entry["action"] == "wait"
    assert entry["result"] == "waiting_codex"
    assert entry["reflection"] == "need more completed runs"


def test_trajectory_memory_limit_is_enforced(tmp_path: Path, monkeypatch) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)
    monkeypatch.setenv("COINT4_TRAJECTORY_MEMORY_MAX", "2")

    state = runner._default_state()
    runner._append_trajectory_memory(state, action="run_next_batch", result="rank_ok", reflection="round 1")
    runner._append_trajectory_memory(state, action="wait", result="waiting_codex", reflection="round 2")
    runner._append_trajectory_memory(state, action="stop", result="stopped_by_codex", reflection="round 3")

    memory = state["trajectory_memory"]
    assert len(memory) == 2
    assert [row["action"] for row in memory] == ["wait", "stop"]


def test_build_decision_context_includes_trajectory_memory_tail(tmp_path: Path) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)
    state = runner._default_state()
    state["trajectory_memory"] = [
        {
            "ts_utc": "2026-02-26T00:00:00Z",
            "iteration": 1,
            "run_group": "rg_a",
            "action": "run_next_batch",
            "result": "rank_ok",
            "reflection": "improved",
        },
        {
            "ts_utc": "2026-02-26T00:01:00Z",
            "iteration": 2,
            "run_group": "rg_b",
            "action": "wait",
            "result": "waiting_codex",
            "reflection": "need data",
        },
    ]

    context = runner._build_decision_context(state)
    tail = context.get("trajectory_memory")

    assert isinstance(tail, list) and len(tail) == 2
    assert tail[-1]["action"] == "wait"
    assert tail[-1]["result"] == "waiting_codex"


def test_decision_prompt_requires_pnl_and_dd_in_human_output(tmp_path: Path) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)
    prompt = runner._decision_prompt({"evidence": {}})
    assert "P&L" in prompt
    assert "DD" in prompt


def test_until_done_keeps_loop_in_process_after_wait(tmp_path: Path, monkeypatch) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)
    runner.args.until_done = True
    runner.args.once = False

    decisions = [
        _decision_payload(
            decision_id="wait-1",
            next_action="wait",
            stop=False,
            stop_reason="need more completed runs",
            human_explanation_md="wait and retry",
            queue_entries=[],
        )
        | {"wait_seconds": 7},
        _decision_payload(
            decision_id="stop-2",
            next_action="stop",
            stop=True,
            stop_reason="finished",
            human_explanation_md="stop now",
            queue_entries=[],
        ),
    ]

    monkeypatch.setattr(runner, "decide_with_codex", _mock_decide_with_codex(module, runner, decisions))
    monkeypatch.setattr(runner, "_quarantine_demo_queues", lambda: None)
    sleep_calls: list[float] = []
    monkeypatch.setattr(module.time, "sleep", lambda sec: sleep_calls.append(float(sec)))

    rc = runner.run()
    assert rc == 0
    assert sleep_calls
    assert int(sleep_calls[0]) == 7

    state = runner.load_state()
    assert state["status"] == "done"
    assert state["last_decision_id"] == "stop-2"

    log_text = runner.main_log_path.read_text(encoding="utf-8")
    assert "continue in-process (--until-done)" in log_text


def test_invalid_codex_json_goes_wait_without_queue(tmp_path: Path, monkeypatch) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)

    runner._last_wait_reason = "CODEX_DECISION_INVALID_JSON:JSONDecodeError"
    monkeypatch.setattr(runner, "decide_with_codex", lambda _state: None)
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


def test_build_decision_context_first_iteration_ignores_rollup_not_updated_with_historical_data(tmp_path: Path) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)

    run_group = "rg_ctx_hist_bootstrap"
    queue_path = runner.app_root / "artifacts" / "wfa" / "aggregate" / run_group / "run_queue.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        "\n".join(
            [
                "config_path,results_dir,status",
                "configs/autopilot/generated/rg_ctx_hist_bootstrap/new_run.yaml,artifacts/wfa/runs_clean/rg_ctx_hist_bootstrap/new_run,planned",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _write_rollup(
        runner.rollup_csv,
        [
            {
                "run_id": "hist_run1",
                "run_group": "rg_ctx_hist_prev",
                "results_dir": "artifacts/wfa/runs_clean/rg_ctx_hist_prev/hist_run1",
                "metrics_path": "hist.csv",
                "config_path": "configs/autopilot/generated/rg_ctx_hist_prev/hist_run1.yaml",
                "status": "completed",
                "metrics_present": "True",
                "sharpe_ratio_abs": "0.80",
                "max_drawdown_on_equity": "-0.10",
            }
        ],
    )

    state = runner._default_state()
    state["iteration"] = 0
    state["current_run_group"] = run_group
    context = runner._build_decision_context(state)
    evidence = context["evidence"]

    assert evidence["historical_bootstrap_mode"] is True
    assert evidence["historical_metrics_present_true_completed"] == 1
    assert "ROLLUP_NOT_UPDATED" not in set(evidence["computed_missing_reasons"])
    assert "ROLLUP_NOT_UPDATED" not in set(evidence["blocking_missing_reasons"])
    assert "NO_COMPLETED_RUNS_CURRENT_QUEUE" in set(evidence["computed_missing_reasons"])
    assert "NO_COMPLETED_RUNS" not in set(evidence["blocking_missing_reasons"])


def test_build_decision_context_after_first_iteration_blocks_rollup_not_updated(tmp_path: Path) -> None:
    module = _load_autonomous_module(tmp_path)
    runner = _setup_runner(module, tmp_path)

    run_group = "rg_ctx_hist_after_first"
    queue_path = runner.app_root / "artifacts" / "wfa" / "aggregate" / run_group / "run_queue.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        "\n".join(
            [
                "config_path,results_dir,status",
                "configs/autopilot/generated/rg_ctx_hist_after_first/new_run.yaml,artifacts/wfa/runs_clean/rg_ctx_hist_after_first/new_run,planned",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _write_rollup(
        runner.rollup_csv,
        [
            {
                "run_id": "hist_run1",
                "run_group": "rg_ctx_hist_prev",
                "results_dir": "artifacts/wfa/runs_clean/rg_ctx_hist_prev/hist_run1",
                "metrics_path": "hist.csv",
                "config_path": "configs/autopilot/generated/rg_ctx_hist_prev/hist_run1.yaml",
                "status": "completed",
                "metrics_present": "True",
                "sharpe_ratio_abs": "0.80",
                "max_drawdown_on_equity": "-0.10",
            }
        ],
    )

    state = runner._default_state()
    state["iteration"] = 1
    state["current_run_group"] = run_group
    context = runner._build_decision_context(state)
    evidence = context["evidence"]

    assert evidence["historical_bootstrap_mode"] is False
    assert "ROLLUP_NOT_UPDATED" in set(evidence["computed_missing_reasons"])
    assert "ROLLUP_NOT_UPDATED" in set(evidence["blocking_missing_reasons"])


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
    monkeypatch.setattr(runner, "decide_with_codex", _mock_decide_with_codex(module, runner, [decision]))
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
