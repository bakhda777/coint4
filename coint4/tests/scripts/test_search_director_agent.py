from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "search_director_agent.py"
SPEC = importlib.util.spec_from_file_location("search_director_agent", SCRIPT_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_build_directive_prefers_winner_proximate_and_yield_tokens() -> None:
    queues = {
        "artifacts/wfa/aggregate/autonomous_seed_a/run_queue.csv": {
            "promotion_verdict": "PROMOTE_PENDING_CONFIRM",
            "strict_pass_count": 1,
            "top_run_group": "strict_rg",
            "rejection_reason": "",
            "strict_gate_reason": "",
        },
        "artifacts/wfa/aggregate/autonomous_seed_b/run_queue.csv": {
            "promotion_verdict": "REJECT",
            "rejection_reason": "METRICS_MISSING",
        },
    }
    yield_state = {
        "active": True,
        "preferred_contains": ["yield_rg"],
        "winner_proximate": {"enabled": True, "contains": ["yield_strict_rg"], "reason": "strict_pass_or_high_yield_lineage"},
        "lane_weights": {"winner_proximate": 40, "broad_search": 45, "confirm_replay": 15},
        "cooldown_contains": ["cold_rg"],
        "preferred_operator_ids": ["op_good"],
        "cooldown_operator_ids": ["op_bad"],
    }

    directive = module.build_directive(queues, yield_state=yield_state)

    assert directive["contains"][:3] == ["strict_rg", "yield_strict_rg", "yield_rg"]
    assert directive["winner_proximate"]["enabled"] is True
    assert directive["yield_governor"]["active"] is True
    assert directive["lane_weights"]["winner_proximate"] == 40


def test_materialize_cold_fail_index_backfills_rejects(tmp_path: Path) -> None:
    cold_path = tmp_path / "cold_fail_index.json"
    queues = {
        "artifacts/wfa/aggregate/q1/run_queue.csv": {
            "promotion_verdict": "REJECT",
            "rejection_reason": "LOW_YIELD_HOMOGENEOUS_METRICS_MISSING",
            "top_run_group": "rg1",
            "candidate_uid": "cand-1",
        },
        "artifacts/wfa/aggregate/q2/run_queue.csv": {
            "promotion_verdict": "ANALYZE",
            "rejection_reason": "",
        },
    }

    summary = module.materialize_cold_fail_index(queues=queues, path=cold_path, ttl_sec=3600)
    payload = json.loads(cold_path.read_text(encoding="utf-8"))

    assert summary["active_count"] == 1
    assert summary["added"] == 1
    assert payload["entries"][0]["queue"] == "artifacts/wfa/aggregate/q1/run_queue.csv"
    assert payload["entries"][0]["gate_reason"] == "METRICS_MISSING"
