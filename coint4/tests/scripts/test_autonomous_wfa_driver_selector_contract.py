from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
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


def test_find_candidate_prefers_executable_pending_over_higher_urgency(tmp_path: Path) -> None:
    app_root = tmp_path
    (app_root / "scripts" / "optimization").mkdir(parents=True, exist_ok=True)
    (app_root / "scripts" / "optimization" / "fullspan_contract.py").write_text(
        FULLSPAN_CONTRACT_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    queue_root = app_root / "artifacts" / "wfa" / "aggregate"
    queue_exec = queue_root / "exec_group" / "run_queue.csv"
    queue_nonexec = queue_root / "nonexec_group" / "run_queue.csv"
    config_exec = app_root / "configs" / "executable.yaml"
    config_exec.parent.mkdir(parents=True, exist_ok=True)
    config_exec.write_text("walk_forward:\n  max_steps: 5\n", encoding="utf-8")

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
        queue_nonexec,
        [
            {
                "config_path": "configs/missing_01.yaml",
                "results_dir": "artifacts/wfa/runs/nonexec_group/run_01",
                "status": "stalled",
            },
            {
                "config_path": "configs/missing_02.yaml",
                "results_dir": "artifacts/wfa/runs/nonexec_group/run_02",
                "status": "stalled",
            },
            {
                "config_path": "configs/missing_03.yaml",
                "results_dir": "artifacts/wfa/runs/nonexec_group/run_03",
                "status": "stalled",
            },
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
        ],
        cwd=app_root,
    )

    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "candidate output must contain one selected queue"
    selected_queue = rows[0]["queue"]

    assert selected_queue == str(queue_exec.relative_to(app_root))
    assert selected_queue != str(queue_nonexec.relative_to(app_root))


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
