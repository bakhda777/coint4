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


def _extract_embedded_python(function_name: str) -> str:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    pattern = rf"{re.escape(function_name)}\(\)\s*\{{.*?<<'PY'\n(?P<body>.*?)\nPY"
    match = re.search(pattern, source, flags=re.DOTALL)
    assert match is not None, f"embedded python for {function_name} not found"
    return match.group("body")


def _run_embedded_python(code: str, argv: list[str], cwd: Path) -> None:
    proc = subprocess.run([sys.executable, "-c", code, *argv], cwd=cwd, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def _write_queue(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config_path", "results_dir", "status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_find_candidate_prefers_planned_over_stalled_only_even_with_higher_urgency(tmp_path: Path) -> None:
    app_root = tmp_path
    (app_root / "scripts" / "optimization").mkdir(parents=True, exist_ok=True)
    (app_root / "scripts" / "optimization" / "fullspan_contract.py").write_text(
        FULLSPAN_CONTRACT_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

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
    (app_root / "scripts" / "optimization").mkdir(parents=True, exist_ok=True)
    (app_root / "scripts" / "optimization" / "fullspan_contract.py").write_text(
        FULLSPAN_CONTRACT_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

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
    (app_root / "scripts" / "optimization").mkdir(parents=True, exist_ok=True)
    (app_root / "scripts" / "optimization" / "fullspan_contract.py").write_text(
        FULLSPAN_CONTRACT_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

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
    assert any(row["queue"] == str(hot_queue.relative_to(app_root)) for row in pool_rows)
