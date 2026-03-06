from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "yield_governor_agent.py"
SPEC = importlib.util.spec_from_file_location("yield_governor_agent", SCRIPT_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_build_yield_governor_state_prefers_strict_and_high_yield(tmp_path: Path) -> None:
    root = tmp_path
    aggregate_dir = root / "artifacts" / "wfa" / "aggregate"
    run_index_path = aggregate_dir / "rollup" / "run_index.csv"
    fullspan_state_path = aggregate_dir / ".autonomous" / "fullspan_decision_state.json"

    queue_dir = aggregate_dir / "autonomous_seed_demo"
    _write_csv(
        queue_dir / "run_queue.csv",
        [
            {
                "config_path": "configs/demo.yaml",
                "results_dir": "artifacts/wfa/runs/demo/holdout_autonomous_seed_demo_v001",
                "status": "completed",
                "lineage_uid": "lineage-good",
                "operator_id": "op_good",
                "metadata_json": "",
            }
        ],
    )
    _write_csv(
        run_index_path,
        [
            {
                "run_id": "holdout_autonomous_seed_demo_v001",
                "run_group": "autonomous_seed_demo",
                "metrics_present": "true",
                "total_trades": "500",
                "total_pairs_traded": "30",
                "max_drawdown_on_equity": "0.05",
                "total_pnl": "100",
                "tail_loss_worst_period_pnl": "-50",
            }
        ],
    )
    fullspan_state_path.parent.mkdir(parents=True, exist_ok=True)
    fullspan_state_path.write_text(
        '{"queues":{"artifacts/wfa/aggregate/demo/run_queue.csv":{"promotion_verdict":"PROMOTE_PENDING_CONFIRM","strict_pass_count":1,"top_run_group":"strict_rg"}}}',
        encoding="utf-8",
    )

    payload = module.build_yield_governor_state(
        root=root,
        aggregate_dir=aggregate_dir,
        run_index_path=run_index_path,
        fullspan_state_path=fullspan_state_path,
        recent_queue_limit=20,
    )

    assert payload["active"] is True
    assert payload["winner_proximate"]["contains"][0] == "strict_rg"
    assert "strict_rg" in payload["preferred_contains"]
