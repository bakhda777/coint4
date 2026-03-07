from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "autonomous_wfa_driver.sh"
FULLSPAN_CONTRACT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "fullspan_contract.py"
SEARCH_QUALITY_CONTRACT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "_search_quality_contract.py"


def _extract_embedded_python(function_name: str) -> str:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    pattern = rf"{re.escape(function_name)}\(\)\s*\{{.*?<<'PY'\n(?P<body>.*?)\nPY"
    match = re.search(pattern, source, flags=re.DOTALL)
    assert match is not None, f"embedded python for {function_name} not found"
    return match.group("body")


def _run_embedded_python(code: str, argv: list[str], cwd: Path) -> None:
    proc = subprocess.run([sys.executable, "-c", code, *argv], cwd=cwd, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def _run_embedded_python_capture(code: str, argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run([sys.executable, "-c", code, *argv], cwd=cwd, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return proc


def _write_queue(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config_path", "results_dir", "status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _prepare_opt_scripts(app_root: Path) -> None:
    opt_dir = app_root / "scripts" / "optimization"
    opt_dir.mkdir(parents=True, exist_ok=True)
    (opt_dir / "fullspan_contract.py").write_text(FULLSPAN_CONTRACT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (opt_dir / "_search_quality_contract.py").write_text(
        SEARCH_QUALITY_CONTRACT_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def test_find_candidate_prefers_planned_over_stalled_only_even_with_higher_urgency(tmp_path: Path) -> None:
    app_root = tmp_path
    _prepare_opt_scripts(app_root)

    queue_root = app_root / "artifacts" / "wfa" / "aggregate"
    queue_exec = queue_root / "exec_group" / "run_queue.csv"
    queue_stalled = queue_root / "stalled_group" / "run_queue.csv"
    config_exec = app_root / "configs" / "executable.yaml"
    config_exec.parent.mkdir(parents=True, exist_ok=True)
    config_exec.write_text("walk_forward:\n  max_steps: 5\n", encoding="utf-8")
    for idx in range(1, 6):
        cfg = app_root / "configs" / f"stalled_{idx}.yaml"
        cfg.write_text("walk_forward:\n  max_steps: 5\n", encoding="utf-8")

    _write_queue(
        queue_exec,
        [
            {
                "config_path": "configs/executable.yaml",
                "results_dir": "artifacts/wfa/runs/exec_group/run_01",
                "status": "planned",
            }
        ],
    )
    _write_queue(
        queue_stalled,
        [
            {
                "config_path": f"configs/stalled_{idx}.yaml",
                "results_dir": f"artifacts/wfa/runs/stalled_group/run_{idx:02d}",
                "status": "stalled",
            }
            for idx in range(1, 6)
        ],
    )

    out_csv = app_root / "candidate.csv"
    code = _extract_embedded_python("find_candidate") + "\nemit_scores()\n"
    _run_embedded_python(
        code,
        [
            str(queue_root),
            str(out_csv),
            "",
            "",
            "8",
            "",
            "0.70",
            "2",
            "2",
            "0.20",
        ],
        cwd=app_root,
    )

    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "candidate output must contain one selected queue"
    selected_queue = rows[0]["queue"]

    assert selected_queue == str(queue_exec.relative_to(app_root))
    assert selected_queue != str(queue_stalled.relative_to(app_root))
    assert rows[0]["effective_planned_count"] == "1"
    assert 0.0 <= float(rows[0]["stalled_share"]) <= 1.0
    float(rows[0]["queue_yield_score"])


def test_find_candidate_penalizes_high_stalled_share_queue(tmp_path: Path) -> None:
    app_root = tmp_path
    _prepare_opt_scripts(app_root)

    queue_root = app_root / "artifacts" / "wfa" / "aggregate"
    queue_high = queue_root / "high_stalled_share" / "run_queue.csv"
    queue_lower = queue_root / "lower_stalled_share" / "run_queue.csv"
    for idx in range(1, 12):
        cfg = app_root / "configs" / f"share_{idx}.yaml"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text("walk_forward:\n  max_steps: 5\n", encoding="utf-8")

    _write_queue(
        queue_high,
        [
            {
                "config_path": f"configs/share_{idx}.yaml",
                "results_dir": f"artifacts/wfa/runs/high_stalled_share/run_{idx:02d}",
                "status": "stalled" if idx <= 5 else "failed",
            }
            for idx in range(1, 11)
        ],
    )
    _write_queue(
        queue_lower,
        [
            {
                "config_path": f"configs/share_{idx}.yaml",
                "results_dir": f"artifacts/wfa/runs/lower_stalled_share/run_{idx:02d}",
                "status": "stalled" if idx <= 4 else ("running" if idx <= 9 else "failed"),
            }
            for idx in range(1, 11)
        ],
    )

    out_csv = app_root / "candidate.csv"
    code = _extract_embedded_python("find_candidate") + "\nemit_scores()\n"
    _run_embedded_python(
        code,
        [
            str(queue_root),
            str(out_csv),
            "",
            "",
            "8",
            "",
            "0.70",
            "2",
            "2",
            "0.20",
        ],
        cwd=app_root,
    )

    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows, "candidate output must contain one selected queue"
    selected_queue = rows[0]["queue"]
    assert selected_queue == str(queue_lower.relative_to(app_root))
    assert float(rows[0]["stalled_share"]) <= 0.5
    assert float(rows[0]["queue_yield_score"]) >= float(rows[0]["pre_rank_score"]) - 1000.0


def test_fallback_pending_candidate_handles_relative_root_skip_forms_and_mtime(tmp_path: Path) -> None:
    app_root = tmp_path
    queue_root = app_root / "artifacts" / "wfa" / "aggregate"
    queue_old = queue_root / "old_group" / "run_queue.csv"
    queue_new = queue_root / "new_group" / "run_queue.csv"

    (app_root / "configs").mkdir(parents=True, exist_ok=True)
    (app_root / "configs" / "old.yaml").write_text("walk_forward:\n  max_steps: 5\n", encoding="utf-8")
    (app_root / "configs" / "new.yaml").write_text("walk_forward:\n  max_steps: 5\n", encoding="utf-8")

    _write_queue(
        queue_old,
        [
            {
                "config_path": "configs/old.yaml",
                "results_dir": "artifacts/wfa/runs/old_group/run_01",
                "status": "planned",
            }
        ],
    )
    _write_queue(
        queue_new,
        [
            {
                "config_path": "configs/new.yaml",
                "results_dir": "artifacts/wfa/runs/new_group/run_01",
                "status": "planned",
            }
        ],
    )

    now = time.time()
    os.utime(queue_old, (now - 3600, now - 3600))
    os.utime(queue_new, (now - 30, now - 30))

    out_csv = app_root / "candidate.csv"
    code = _extract_embedded_python("fallback_pending_candidate")
    rel_queue_root = str(queue_root.relative_to(app_root))

    def _select(queue_root_arg: str, skip_queue: str) -> dict[str, str]:
        _run_embedded_python(
            code,
            [
                queue_root_arg,
                str(out_csv),
                "",
                skip_queue,
                "",
            ],
            cwd=app_root,
        )
        with out_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, "fallback candidate output must contain one selected queue"
        return rows[0]

    first = _select(rel_queue_root, "")
    first_queue = first["queue"]
    assert first_queue
    assert not Path(first_queue).is_absolute()
    assert int(first["mtime"]) == int((app_root / first_queue).stat().st_mtime)

    second = _select(rel_queue_root, first_queue)
    second_queue = second["queue"]
    assert second_queue != first_queue
    assert int(second["mtime"]) == int((app_root / second_queue).stat().st_mtime)

    second_abs = str((app_root / second_queue).resolve())
    third = _select(str(queue_root), second_abs)
    third_queue = third["queue"]
    assert third_queue == first_queue

    fourth = _select(str(queue_root), "/" + third_queue.lstrip("/"))
    assert fourth["queue"] == second_queue


def test_log_decision_note_writes_contract_fields_and_keeps_compat(tmp_path: Path) -> None:
    code = _extract_embedded_python("log_decision_note")
    notes_path = tmp_path / "decision_notes.jsonl"

    _run_embedded_python(
        code,
        [
            str(notes_path),
            "artifacts/wfa/aggregate/demo/run_queue.csv",
            "REJECT",
            "gate_status=HARD_FAIL reason=DD_FAIL",
            "skip_and_select_next_candidate",
            "2026-03-05T00:00:00Z",
            "fullspan_v1",
            "fullspan",
            "promote_profile",
            "REJECT",
            "HARD_FAIL",
            "DD_FAIL",
            "-10000",
            "score_fullspan_v1",
        ],
        cwd=tmp_path,
    )
    _run_embedded_python(
        code,
        [
            str(notes_path),
            "global",
            "VPS_FORCE_CYCLE_FAIL",
            "reason=vps_unreachable",
            "await_next_watchdog_cycle",
            "2026-03-05T00:00:01Z",
        ],
        cwd=tmp_path,
    )

    records = [json.loads(line) for line in notes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 2

    rich = records[0]
    assert rich["queue"] == "artifacts/wfa/aggregate/demo/run_queue.csv"
    assert rich["selection_policy"] == "fullspan_v1"
    assert rich["selection_mode"] == "fullspan"
    assert rich["selection_profile"] == "promote_profile"
    assert rich["promotion_verdict"] == "REJECT"
    assert rich["gate_status"] == "HARD_FAIL"
    assert rich["gate_reason"] == "DD_FAIL"
    assert rich["pre_rank_score"] == "-10000"
    assert rich["ranking_primary_key"] == "score_fullspan_v1"

    compat = records[1]
    assert compat["queue"] == "global"
    assert compat["selection_policy"] == ""
    assert compat["selection_mode"] == ""
    assert compat["selection_profile"] == ""
    assert compat["promotion_verdict"] == ""
    assert compat["gate_status"] == ""
    assert compat["gate_reason"] == ""
    assert compat["pre_rank_score"] == ""
    assert compat["ranking_primary_key"] == ""


def test_queue_start_confirmation_status_classifies_manifest_mismatch(tmp_path: Path) -> None:
    app_root = tmp_path
    queue_rel = "artifacts/wfa/aggregate/demo/run_queue.csv"
    queue_path = app_root / queue_rel
    _write_queue(
        queue_path,
        [
            {
                "config_path": "configs/demo.yaml",
                "results_dir": "artifacts/wfa/runs/demo/run_01",
                "status": "planned",
            }
        ],
    )
    qlog = app_root / "startup.log"
    qlog.write_text(
        "\n".join(
            [
                "powered: sync_code failed error_class=REMOTE_SYNC_FAILED fatal=true msg=sync_code verification failed: remote manifest mismatch drift_detected=true drift_reason=manifest_mismatch",
                "powered: FAIL reason=REMOTE_SYNC_FAILED",
            ]
        ),
        encoding="utf-8",
    )

    code = _extract_embedded_python("queue_start_confirmation_status")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(app_root),
            queue_rel,
            str(qlog),
            str(int(time.time())),
            "180",
            "120",
            "420",
        ],
        cwd=app_root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.stdout.strip() == "fail\tMANIFEST_MISMATCH\tmanifest_mismatch"


def test_trigger_confirm_fastlane_shortlist_excludes_stress_rows(tmp_path: Path) -> None:
    source_queue = tmp_path / "source_run_queue.csv"
    shortlist = tmp_path / "shortlist.csv"
    fallback_shortlist = tmp_path / "shortlist_fallback.csv"

    _write_queue(
        source_queue,
        [
            {
                "config_path": "configs/target_stress.yaml",
                "results_dir": "artifacts/wfa/runs/group/holdout_target_stress",
                "status": "completed",
            },
            {
                "config_path": "configs/stress_target.yaml",
                "results_dir": "artifacts/wfa/runs/group/holdout_target_pref",
                "status": "completed",
            },
            {
                "config_path": "configs/target_by_runid.yaml",
                "results_dir": "artifacts/wfa/runs/group/stress_target_by_runid",
                "status": "completed",
            },
            {
                "config_path": "configs/target_by_dir.yaml",
                "results_dir": "artifacts/wfa/runs/group/stress/target_by_dir",
                "status": "completed",
            },
            {
                "config_path": "configs/target.yaml",
                "results_dir": "artifacts/wfa/runs/group/holdout_target",
                "status": "completed",
            },
        ],
    )

    code = _extract_embedded_python("trigger_confirm_fastlane")
    _run_embedded_python(
        code,
        [str(source_queue), str(shortlist), "target"],
        cwd=tmp_path,
    )

    with shortlist.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["config_path"] == "configs/target.yaml"
    assert rows[0]["results_dir"] == "artifacts/wfa/runs/group/holdout_target"

    _run_embedded_python(
        code,
        [str(source_queue), str(fallback_shortlist), "missing_variant"],
        cwd=tmp_path,
    )
    with fallback_shortlist.open(newline="", encoding="utf-8") as handle:
        fallback_rows = list(csv.DictReader(handle))

    assert len(fallback_rows) == 1
    assert fallback_rows[0]["config_path"] == "configs/target.yaml"
    assert fallback_rows[0]["results_dir"] == "artifacts/wfa/runs/group/holdout_target"


def test_find_candidate_skips_active_cold_fail_and_writes_pool(tmp_path: Path) -> None:
    app_root = tmp_path
    _prepare_opt_scripts(app_root)

    queue_root = app_root / "artifacts" / "wfa" / "aggregate"
    cold_queue = queue_root / "cold_group" / "run_queue.csv"
    hot_queue = queue_root / "hot_group" / "run_queue.csv"
    pool_csv = app_root / "candidate_pool.csv"
    cold_state = app_root / "cold_fail_index.json"

    (app_root / "configs").mkdir(parents=True, exist_ok=True)
    (app_root / "configs" / "cold.yaml").write_text("walk_forward:\n  max_steps: 5\n", encoding="utf-8")
    (app_root / "configs" / "hot.yaml").write_text("walk_forward:\n  max_steps: 5\n", encoding="utf-8")

    _write_queue(
        cold_queue,
        [
            {
                "config_path": "configs/cold.yaml",
                "results_dir": "artifacts/wfa/runs/cold_group/run_01",
                "status": "planned",
            }
        ],
    )
    _write_queue(
        hot_queue,
        [
            {
                "config_path": "configs/hot.yaml",
                "results_dir": "artifacts/wfa/runs/hot_group/run_01",
                "status": "planned",
            }
        ],
    )

    cold_mtime = int(cold_queue.stat().st_mtime)
    cold_state.write_text(
        json.dumps(
            {
                "policy_version": "fullspan_v1",
                "entries": [
                    {
                        "queue": str(cold_queue.relative_to(app_root)),
                        "gate_reason": "DD_FAIL",
                        "inserted_ts": cold_mtime + 1,
                        "until_ts": cold_mtime + 7200,
                        "policy_version": "fullspan_v1",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out_csv = app_root / "candidate.csv"
    code = _extract_embedded_python("find_candidate") + "\nemit_scores()\n"
    _run_embedded_python(
        code,
        [
            str(queue_root),
            str(out_csv),
            "",
            "",
            "8",
            "",
            "0.70",
            "2",
            "2",
            "0.20",
            str(pool_csv),
            str(cold_state),
            "3",
            "fullspan_v1",
        ],
        cwd=app_root,
    )

    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["queue"] == str(hot_queue.relative_to(app_root))

    with pool_csv.open(newline="", encoding="utf-8") as handle:
        pool_rows = list(csv.DictReader(handle))
    assert pool_rows
    assert all(row["queue"] != str(cold_queue.relative_to(app_root)) for row in pool_rows)
    hot_row = next(row for row in pool_rows if row["queue"] == str(hot_queue.relative_to(app_root)))
    assert int(hot_row["dispatchable_pending"]) == 1
    assert int(hot_row["executable_pending"]) == 1


def test_find_candidate_skips_fail_closed_queue_state(tmp_path: Path) -> None:
    app_root = tmp_path
    _prepare_opt_scripts(app_root)

    queue_root = app_root / "artifacts" / "wfa" / "aggregate"
    blocked_queue = queue_root / "blocked_group" / "run_queue.csv"
    open_queue = queue_root / "open_group" / "run_queue.csv"
    fullspan_state = app_root / "fullspan_state.json"

    (app_root / "configs").mkdir(parents=True, exist_ok=True)
    (app_root / "configs" / "blocked.yaml").write_text("walk_forward:\n  max_steps: 5\n", encoding="utf-8")
    (app_root / "configs" / "open.yaml").write_text("walk_forward:\n  max_steps: 5\n", encoding="utf-8")

    _write_queue(
        blocked_queue,
        [
            {
                "config_path": "configs/blocked.yaml",
                "results_dir": "artifacts/wfa/runs/blocked_group/run_01",
                "status": "planned",
            }
        ],
    )
    _write_queue(
        open_queue,
        [
            {
                "config_path": "configs/open.yaml",
                "results_dir": "artifacts/wfa/runs/open_group/run_01",
                "status": "planned",
            }
        ],
    )

    fullspan_state.write_text(
        json.dumps(
            {
                "queues": {
                    str(blocked_queue.relative_to(app_root)): {
                        "promotion_verdict": "ANALYZE",
                        "cutover_permission": "FAIL_CLOSED",
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    out_csv = app_root / "candidate.csv"
    code = _extract_embedded_python("find_candidate") + "\nemit_scores()\n"
    _run_embedded_python(
        code,
        [
            str(queue_root),
            str(out_csv),
            "",
            "",
            "8",
            str(fullspan_state),
            "0.70",
            "2",
            "2",
            "0.20",
        ],
        cwd=app_root,
    )

    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert rows[0]["queue"] == str(open_queue.relative_to(app_root))


def test_ready_buffer_policy_hash_and_replay_fastlane_hooks_contract() -> None:
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    required = [
        "current_planner_policy_hash()",
        "ready_buffer_policy_hash()",
        "ready_buffer_snapshot_hash()",
        "planner_policy_hash",
        "snapshot_hash",
        "queue_file_mtime",
        "ready_buffer_policy_mismatch_count",
        "dispatch_replay_fastlane_hooks()",
        "REPLAY_FASTLANE_SCAN_LIMIT",
        "fastlane_replay_pending",
        "winner_parent_duplication_rate",
    ]
    missing = [snippet for snippet in required if snippet not in src]
    assert not missing, f"missing runtime selector hooks: {missing}"


def test_ready_buffer_snapshot_hash_changes_without_policy_mismatch(tmp_path: Path) -> None:
    policy_code = _extract_embedded_python("ready_buffer_policy_hash")
    snapshot_code = _extract_embedded_python("ready_buffer_snapshot_hash")
    pool_csv = tmp_path / "candidate_pool.csv"
    pool_csv.write_text(
        "queue,planned,running,stalled,failed,completed,total,urgency,mtime,promotion_potential,gate_status,"
        "gate_reason,pre_rank_score,strict_gate_status,strict_gate_reason,effective_planned_count,"
        "stalled_share,queue_yield_score,recent_yield,dispatchable_pending,executable_pending\n"
        "artifacts/wfa/aggregate/demo/run_queue.csv,1,0,0,0,0,1,1.000,123,POSSIBLE,OPEN,seed,5.000,"
        "FULLSPAN_PREFILTER_PASSED,,1,0.000000,5.000,0.000,1,1\n",
        encoding="utf-8",
    )
    surrogate_path = tmp_path / "surrogate.json"
    fullspan_path = tmp_path / "fullspan.json"
    surrogate_path.write_text("{}", encoding="utf-8")
    fullspan_path.write_text("{}", encoding="utf-8")
    planner_policy_hash = "planner-A"

    policy_one = _run_embedded_python_capture(
        policy_code,
        ["fullspan_v1", "3", "2", planner_policy_hash],
        cwd=tmp_path,
    ).stdout.strip()
    snapshot_one = _run_embedded_python_capture(
        snapshot_code,
        [str(pool_csv), str(surrogate_path), str(fullspan_path), "3", planner_policy_hash],
        cwd=tmp_path,
    ).stdout.strip()

    time.sleep(1)
    pool_csv.write_text(pool_csv.read_text(encoding="utf-8").replace(",123,", ",124,"), encoding="utf-8")

    policy_two = _run_embedded_python_capture(
        policy_code,
        ["fullspan_v1", "3", "2", planner_policy_hash],
        cwd=tmp_path,
    ).stdout.strip()
    snapshot_two = _run_embedded_python_capture(
        snapshot_code,
        [str(pool_csv), str(surrogate_path), str(fullspan_path), "3", planner_policy_hash],
        cwd=tmp_path,
    ).stdout.strip()

    assert policy_one == policy_two
    assert snapshot_one != snapshot_two


def test_early_stop_low_yield_queue_marks_min_windows_unreachable(tmp_path: Path) -> None:
    queue_rel = "artifacts/wfa/aggregate/demo/run_queue.csv"
    queue_path = tmp_path / queue_rel
    run_index_path = tmp_path / "artifacts" / "wfa" / "aggregate" / "rollup" / "run_index.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    run_index_path.parent.mkdir(parents=True, exist_ok=True)

    _write_queue(
        queue_path,
        [
            {
                "config_path": "configs/alpha.yaml",
                "results_dir": "artifacts/wfa/runs/demo/holdout_alpha_oos1",
                "status": "completed",
            },
            {
                "config_path": "configs/beta.yaml",
                "results_dir": "artifacts/wfa/runs/demo/holdout_beta_oos1",
                "status": "planned",
            },
        ],
    )
    with run_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "status",
                "metrics_present",
                "coverage_ratio",
                "total_trades",
                "total_pairs_traded",
                "max_drawdown_on_equity",
                "total_pnl",
                "tail_loss_worst_period_pnl",
                "tail_loss_worst_pair_pnl",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "holdout_alpha_oos1",
                "status": "completed",
                "metrics_present": "1",
                "coverage_ratio": "1.0",
                "total_trades": "500",
                "total_pairs_traded": "50",
                "max_drawdown_on_equity": "0.05",
                "total_pnl": "100.0",
                "tail_loss_worst_period_pnl": "-50.0",
                "tail_loss_worst_pair_pnl": "-25.0",
            }
        )

    code = _extract_embedded_python("early_stop_low_yield_queue")
    proc = _run_embedded_python_capture(
        code,
        [
            queue_rel,
            str(queue_path),
            str(run_index_path),
            "8",
            "0.75",
            "0.70",
            "6",
            "12",
            "0.80",
            "6",
            "3",
            "0.95",
        ],
        cwd=tmp_path,
    )
    payload = json.loads(proc.stdout.strip())
    assert payload["trigger"] is True
    assert payload["reason"] == "MIN_WINDOWS_UNREACHABLE"
    assert payload["changed"] == 1

    with queue_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "completed"
    assert rows[1]["status"] == "skipped"


def test_early_stop_low_yield_queue_marks_coverage_unreachable(tmp_path: Path) -> None:
    queue_rel = "artifacts/wfa/aggregate/demo/run_queue.csv"
    queue_path = tmp_path / queue_rel
    run_index_path = tmp_path / "artifacts" / "wfa" / "aggregate" / "rollup" / "run_index.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    run_index_path.parent.mkdir(parents=True, exist_ok=True)

    _write_queue(
        queue_path,
        [
            {
                "config_path": "configs/alpha.yaml",
                "results_dir": "artifacts/wfa/runs/demo/holdout_alpha_oos1",
                "status": "completed",
            },
            {
                "config_path": "configs/alpha.yaml",
                "results_dir": "artifacts/wfa/runs/demo/holdout_alpha_oos2",
                "status": "planned",
            },
        ],
    )
    with run_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "status",
                "metrics_present",
                "coverage_ratio",
                "total_trades",
                "total_pairs_traded",
                "max_drawdown_on_equity",
                "total_pnl",
                "tail_loss_worst_period_pnl",
                "tail_loss_worst_pair_pnl",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "holdout_alpha_oos1",
                "status": "completed",
                "metrics_present": "1",
                "coverage_ratio": "0.40",
                "total_trades": "500",
                "total_pairs_traded": "50",
                "max_drawdown_on_equity": "0.05",
                "total_pnl": "100.0",
                "tail_loss_worst_period_pnl": "-50.0",
                "tail_loss_worst_pair_pnl": "-25.0",
            }
        )

    code = _extract_embedded_python("early_stop_low_yield_queue")
    proc = _run_embedded_python_capture(
        code,
        [
            queue_rel,
            str(queue_path),
            str(run_index_path),
            "8",
            "0.75",
            "0.70",
            "6",
            "12",
            "0.80",
            "6",
            "2",
            "0.95",
        ],
        cwd=tmp_path,
    )
    payload = json.loads(proc.stdout.strip())
    assert payload["trigger"] is True
    assert payload["reason"] == "COVERAGE_UNREACHABLE"
    assert payload["changed"] == 1

    with queue_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "completed"
    assert rows[1]["status"] == "skipped"


def test_early_stop_low_yield_queue_skips_controlled_recovery_queue(tmp_path: Path) -> None:
    queue_rel = "artifacts/wfa/aggregate/demo/run_queue.csv"
    queue_path = tmp_path / queue_rel
    run_index_path = tmp_path / "artifacts" / "wfa" / "aggregate" / "rollup" / "run_index.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    run_index_path.parent.mkdir(parents=True, exist_ok=True)

    _write_queue(
        queue_path,
        [
            {
                "config_path": "configs/alpha.yaml",
                "results_dir": "artifacts/wfa/runs/demo/holdout_alpha_oos1",
                "status": "planned",
            },
            {
                "config_path": "configs/beta.yaml",
                "results_dir": "artifacts/wfa/runs/demo/holdout_beta_oos1",
                "status": "planned",
            },
        ],
    )
    (queue_path.parent / "queue_policy.json").write_text(
        json.dumps({"recovery_mode": "controlled"}, ensure_ascii=False),
        encoding="utf-8",
    )
    with run_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "status",
                "metrics_present",
                "coverage_ratio",
                "total_trades",
                "total_pairs_traded",
                "max_drawdown_on_equity",
                "total_pnl",
                "tail_loss_worst_period_pnl",
                "tail_loss_worst_pair_pnl",
            ],
        )
        writer.writeheader()

    code = _extract_embedded_python("early_stop_low_yield_queue")
    proc = _run_embedded_python_capture(
        code,
        [
            queue_rel,
            str(queue_path),
            str(run_index_path),
            "8",
            "0.75",
            "0.70",
            "6",
            "12",
            "0.80",
            "6",
            "3",
            "0.95",
        ],
        cwd=tmp_path,
    )
    payload = json.loads(proc.stdout.strip())
    assert payload["trigger"] is False
    assert payload["recovery_mode"] == "controlled"

    with queue_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["planned", "planned"]
