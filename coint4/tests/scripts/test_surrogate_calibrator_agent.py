from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "surrogate_calibrator_agent.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_surrogate_calibrator_applies_hysteresis_clamped_thresholds(tmp_path: Path) -> None:
    app_root = tmp_path
    state_dir = app_root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    run_index_path = app_root / "artifacts" / "wfa" / "aggregate" / "rollup" / "run_index.csv"
    run_index_path.parent.mkdir(parents=True, exist_ok=True)
    run_index_path.write_text("run_group,status,metrics_present\n", encoding="utf-8")

    gate_queues = {}
    fullspan_queues = {}
    for idx in range(60):
        queue = f"artifacts/wfa/aggregate/pos_{idx:03d}/run_queue.csv"
        gate_queues[queue] = {"risk_score": 0.10 + idx * 0.002, "decision": "allow"}
        fullspan_queues[queue] = {"strict_pass_count": 1, "promotion_verdict": "PROMOTE_PENDING_CONFIRM"}
    for idx in range(60):
        queue = f"artifacts/wfa/aggregate/neg_{idx:03d}/run_queue.csv"
        gate_queues[queue] = {"risk_score": 0.80 + idx * 0.002, "decision": "reject"}
        fullspan_queues[queue] = {
            "strict_pass_count": 0,
            "promotion_verdict": "REJECT",
            "rejection_reason": "NO_PROGRESS",
            "strict_gate_reason": "NO_PROGRESS",
        }

    _write_json(
        state_dir / "gate_surrogate_state.json",
        {
            "hard_fail_risk_policy": {"reject_threshold": 0.75, "refine_threshold": 0.45},
            "queues": gate_queues,
        },
    )
    _write_json(state_dir / "fullspan_decision_state.json", {"queues": fullspan_queues, "runtime_metrics": {}})
    (state_dir / "decision_notes.jsonl").write_text("", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--root", str(app_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads((state_dir / "surrogate_calibration_state.json").read_text(encoding="utf-8"))
    assert payload["enabled"] is True
    assert payload["applied"] is True
    assert payload["sample_size"] == 120
    assert payload["applied_reject_threshold"] == 0.8
    assert 0.45 <= payload["applied_refine_threshold"] <= 0.5
