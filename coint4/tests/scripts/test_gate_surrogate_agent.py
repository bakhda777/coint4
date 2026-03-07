from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "gate_surrogate_agent.py"
SPEC = importlib.util.spec_from_file_location("gate_surrogate_agent", SCRIPT_PATH)
assert SPEC and SPEC.loader
gate_surrogate_agent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate_surrogate_agent)


def _run_index_pending_only(rows: int = 3) -> dict[str, object]:
    return {
        "rows": rows,
        "status": {"planned": rows},
        "metrics_missing": 0,
        "completed": 0,
        "completed_zero_activity": 0,
        "completed_informative": 0,
        "lineages": {},
        "legacy_lineages": {},
    }


def test_decide_queue_allows_clean_cold_start_dispatch() -> None:
    payload, lineages, validation_error = gate_surrogate_agent.decide_queue(
        queue_key="artifacts/wfa/aggregate/autonomous_seed_demo/run_queue.csv",
        queue_rows=8,
        queue_status={"planned": 8},
        run_group="autonomous_seed_demo",
        run_index_entry=None,
        fullspan_entry=None,
        quarantine_entry=None,
        quarantine_active=False,
        reject_threshold=0.75,
        refine_threshold=0.45,
    )

    assert payload["decision"] == "allow"
    assert payload["reason"] == "cold_start_clean_queue"
    assert payload["risk_score"] == 0.3
    assert payload["evidence"]["cold_start_exploration"] is True
    assert lineages == []
    assert validation_error is False


def test_decide_queue_keeps_refine_for_dirty_cold_start_queue() -> None:
    payload, _lineages, _validation_error = gate_surrogate_agent.decide_queue(
        queue_key="artifacts/wfa/aggregate/autonomous_seed_demo_dirty/run_queue.csv",
        queue_rows=5,
        queue_status={"planned": 4, "failed": 1},
        run_group="autonomous_seed_demo_dirty",
        run_index_entry=None,
        fullspan_entry=None,
        quarantine_entry=None,
        quarantine_active=False,
        reject_threshold=0.75,
        refine_threshold=0.45,
    )

    assert payload["decision"] == "refine"
    assert payload["reason"] == "queue_pending_backlog"
    assert payload["risk_score"] == 0.6
    assert payload["evidence"]["cold_start_exploration"] is False


def test_decide_queue_allows_placeholder_fullspan_probe_for_fresh_queue() -> None:
    payload, _lineages, _validation_error = gate_surrogate_agent.decide_queue(
        queue_key="artifacts/wfa/aggregate/autonomous_seed_live_like/run_queue.csv",
        queue_rows=12,
        queue_status={"planned": 10, "skipped": 2},
        run_group="autonomous_seed_live_like",
        run_index_entry=None,
        fullspan_entry={
            "promotion_verdict": "ANALYZE",
            "contract_reason": "METRICS_MISSING",
            "contract_windows_total": 0,
            "contract_windows_passed": 0,
            "score_fullspan_v1": None,
            "avg_robust_sharpe": None,
        },
        quarantine_entry=None,
        quarantine_active=True,
        reject_threshold=0.75,
        refine_threshold=0.45,
    )

    assert payload["decision"] == "allow"
    assert payload["reason"] == "cold_start_clean_queue"
    assert payload["risk_score"] == 0.31
    assert payload["evidence"]["cold_start_exploration"] is True
    assert payload["evidence"]["fullspan_placeholder_probe"] is True


def test_decide_queue_idle_slot_override_allows_fresh_backlog_queue() -> None:
    payload, _lineages, _validation_error = gate_surrogate_agent.decide_queue(
        queue_key="artifacts/wfa/aggregate/autonomous_seed_idle_slot/run_queue.csv",
        queue_rows=10,
        queue_status={"planned": 10},
        run_group="autonomous_seed_idle_slot",
        run_index_entry=_run_index_pending_only(),
        fullspan_entry=None,
        quarantine_entry=None,
        quarantine_active=False,
        reject_threshold=0.75,
        refine_threshold=0.45,
        idle_slot_available=True,
        idle_slot_source="process_slo_idle_with_executable_pending",
    )

    assert payload["decision"] == "allow"
    assert payload["reason"] == "cold_start_idle_slot"
    assert payload["risk_score"] < 0.45
    assert payload["evidence"]["fresh_clean_queue"] is True
    assert payload["evidence"]["idle_slot_available"] is True
    assert payload["evidence"]["idle_slot_source"] == "process_slo_idle_with_executable_pending"
    assert payload["evidence"]["cold_start_idle_slot_override"] is True
    assert any(item["key"] == "cold_start_idle_slot" for item in payload["evidence"]["contributions"])


def test_decide_queue_without_idle_slot_keeps_refine_for_fresh_backlog_queue() -> None:
    payload, _lineages, _validation_error = gate_surrogate_agent.decide_queue(
        queue_key="artifacts/wfa/aggregate/autonomous_seed_idle_slot/run_queue.csv",
        queue_rows=10,
        queue_status={"planned": 10},
        run_group="autonomous_seed_idle_slot",
        run_index_entry=_run_index_pending_only(),
        fullspan_entry=None,
        quarantine_entry=None,
        quarantine_active=False,
        reject_threshold=0.75,
        refine_threshold=0.45,
        idle_slot_available=False,
        idle_slot_source="",
    )

    assert payload["decision"] == "refine"
    assert payload["reason"] == "queue_pending_backlog"
    assert payload["risk_score"] >= 0.45
    assert payload["evidence"]["fresh_clean_queue"] is True
    assert payload["evidence"]["cold_start_idle_slot_override"] is False


def test_decide_queue_skipped_only_queue_does_not_trigger_skipped_dominant_refine() -> None:
    payload, _lineages, _validation_error = gate_surrogate_agent.decide_queue(
        queue_key="artifacts/wfa/aggregate/autonomous_seed_skipped/run_queue.csv",
        queue_rows=8,
        queue_status={"skipped": 8},
        run_group="autonomous_seed_skipped",
        run_index_entry=None,
        fullspan_entry={
            "promotion_verdict": "ANALYZE",
            "contract_reason": "METRICS_MISSING",
            "contract_windows_total": 0,
            "contract_windows_passed": 0,
        },
        quarantine_entry=None,
        quarantine_active=True,
        reject_threshold=0.75,
        refine_threshold=0.45,
    )

    assert payload["decision"] == "allow"
    assert payload["reason"] != "queue_skipped_dominant"
    assert payload["risk_score"] < 0.45


def test_decide_queue_keeps_skipped_dominant_when_pending_work_remains() -> None:
    payload, _lineages, _validation_error = gate_surrogate_agent.decide_queue(
        queue_key="artifacts/wfa/aggregate/autonomous_seed_mixed/run_queue.csv",
        queue_rows=10,
        queue_status={"planned": 1, "skipped": 9},
        run_group="autonomous_seed_mixed",
        run_index_entry={
            "rows": 10,
            "status": {"planned": 1, "skipped": 9},
            "metrics_missing": 0,
            "completed": 0,
            "completed_zero_activity": 0,
            "completed_informative": 0,
        },
        fullspan_entry=None,
        quarantine_entry=None,
        quarantine_active=False,
        reject_threshold=0.75,
        refine_threshold=0.45,
    )

    assert payload["decision"] == "refine"
    assert payload["reason"] == "queue_skipped_dominant"


def test_resolve_idle_slot_signal_prefers_process_slo_flag() -> None:
    signal = gate_surrogate_agent.resolve_idle_slot_signal(
        {
            "queue": {
                "idle_with_executable_pending": True,
                "local_runner_count": 0,
                "remote_runner_count": 0,
                "remote_reachable": True,
            }
        },
        queue_status_map={
            "artifacts/wfa/aggregate/demo/run_queue.csv": (3, {"running": 1, "planned": 2}),
        },
    )

    assert signal["idle_slot_available"] is True
    assert signal["source"] == "process_slo_idle_with_executable_pending"


def test_resolve_idle_slot_signal_uses_safe_fallback_on_missing_state() -> None:
    signal = gate_surrogate_agent.resolve_idle_slot_signal(
        None,
        queue_status_map={
            "artifacts/wfa/aggregate/demo_a/run_queue.csv": (5, {"planned": 5}),
            "artifacts/wfa/aggregate/demo_b/run_queue.csv": (2, {"completed": 2}),
        },
    )

    assert signal["idle_slot_available"] is True
    assert signal["source"] == "fallback_queue_status_idle_slot"
