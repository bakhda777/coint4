from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "gate_surrogate_agent.py"
SPEC = importlib.util.spec_from_file_location("gate_surrogate_agent", SCRIPT_PATH)
assert SPEC and SPEC.loader
gate_surrogate_agent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate_surrogate_agent)


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
