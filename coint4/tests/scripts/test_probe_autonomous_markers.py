from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_probe_module(tmp_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "probe_autonomous_markers.py"
    spec = importlib.util.spec_from_file_location(tmp_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[tmp_name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_collect_markers_reports_surrogate_evidence_present(tmp_path: Path) -> None:
    module = _load_probe_module("probe_markers_evidence_present")
    state_dir = tmp_path / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    now_ts = module.utc_now_iso()

    _write_json(
        state_dir / "fullspan_decision_state.json",
        {
            "queues": {
                "artifacts/wfa/aggregate/demo/run_queue.csv": {
                    "cutover_permission": "FAIL_CLOSED",
                }
            },
            "runtime_metrics": {
                "surrogate_reject_count": 1,
                "surrogate_refine_count": 0,
                "surrogate_allow_count": 1,
            },
        },
    )
    _write_json(
        state_dir / "gate_surrogate_state.json",
        {
            "ts": now_ts,
            "summary": {"mode": "active", "decision_counts": {"allow": 1, "refine": 1, "reject": 1}},
            "queues": {},
            "inputs": {"missing_required_inputs": []},
        },
    )
    _write_json(
        state_dir / "search_director_directive.json",
        {
            "ts": now_ts,
            "mode": "neutral",
            "gate_surrogate_mode": "active",
            "gate_surrogate_ts": now_ts,
        },
    )
    (state_dir / "decision_notes.jsonl").write_text(
        json.dumps(
            {
                "ts": now_ts,
                "queue": "artifacts/wfa/aggregate/demo/run_queue.csv",
                "action": "SURROGATE_REJECT",
                "reason": "reason=demo",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (state_dir / "driver.log").write_text(
        f"{now_ts} | surrogate_gate queue=artifacts/wfa/aggregate/demo/run_queue.csv action=SURROGATE_ALLOW reason=healthy_signal\n",
        encoding="utf-8",
    )

    markers = module.collect_markers(
        fullspan_state_path=state_dir / "fullspan_decision_state.json",
        process_slo_state_path=state_dir / "process_slo_state.json",
        decision_notes_path=state_dir / "decision_notes.jsonl",
        driver_log_path=state_dir / "driver.log",
        gate_surrogate_state_path=state_dir / "gate_surrogate_state.json",
        directive_overlay_path=state_dir / "search_director_directive.json",
    )

    surrogate = markers["surrogate_runtime"]
    assert surrogate["gate_surrogate"]["mode"] == "active"
    assert surrogate["evidence"]["combined"]["SURROGATE_REJECT"]["count"] == 1
    assert surrogate["evidence"]["combined"]["SURROGATE_ALLOW"]["count"] == 1
    assert surrogate["branch_health"]["status"] == "evidence_present"
    assert surrogate["branch_health"]["broken_branch"] is False


def test_collect_markers_flags_broken_branch_when_gate_is_fresh_but_no_evidence(tmp_path: Path) -> None:
    module = _load_probe_module("probe_markers_broken_branch")
    state_dir = tmp_path / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    now_ts = module.utc_now_iso()

    _write_json(state_dir / "fullspan_decision_state.json", {"queues": {}, "runtime_metrics": {}})
    _write_json(
        state_dir / "search_director_directive.json",
        {
            "ts": now_ts,
            "mode": "neutral",
            "gate_surrogate_mode": "active",
            "gate_surrogate_ts": now_ts,
        },
    )
    eligible_queue = "artifacts/wfa/aggregate/demo_refine/run_queue.csv"
    _write_json(
        state_dir / "gate_surrogate_state.json",
        {
            "ts": now_ts,
            "summary": {"mode": "active", "decision_counts": {"allow": 0, "refine": 2, "reject": 1}},
            "queues": {
                eligible_queue: {"decision": "refine", "risk_score": 0.6},
            },
            "inputs": {"missing_required_inputs": []},
        },
    )
    (state_dir / "decision_notes.jsonl").write_text("", encoding="utf-8")
    (state_dir / "driver.log").write_text(
        f"{now_ts} | candidate queue={eligible_queue} reason=planned_work cause=UNKNOWN\n",
        encoding="utf-8",
    )

    markers = module.collect_markers(
        fullspan_state_path=state_dir / "fullspan_decision_state.json",
        process_slo_state_path=state_dir / "process_slo_state.json",
        decision_notes_path=state_dir / "decision_notes.jsonl",
        driver_log_path=state_dir / "driver.log",
        gate_surrogate_state_path=state_dir / "gate_surrogate_state.json",
        directive_overlay_path=state_dir / "search_director_directive.json",
    )

    surrogate = markers["surrogate_runtime"]
    assert surrogate["branch_health"]["status"] == "broken_branch"
    assert surrogate["branch_health"]["broken_branch"] is True
