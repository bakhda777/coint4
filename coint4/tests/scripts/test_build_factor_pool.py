from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_script(tmp_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts/optimization/build_factor_pool.py"
    spec = importlib.util.spec_from_file_location(f"build_factor_pool_test_{tmp_name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_run_index(path: Path, *, run_group: str, candidate_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "run_group",
        "config_path",
        "results_dir",
        "status",
        "metrics_present",
        "sharpe_ratio_abs",
        "max_drawdown_on_equity",
        "total_trades",
        "total_pairs_traded",
        "wf_zero_pair_steps_pct",
        "tail_loss_worst_pair_share",
        "tail_loss_worst_period_share",
    ]
    base_id = f"{run_group}_v001_{candidate_id}_oos20220101_20221231"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for kind, sharpe, dd in [("holdout", "1.2", "-0.08"), ("stress", "1.0", "-0.10")]:
            writer.writerow(
                {
                    "run_id": f"{kind}_{base_id}",
                    "run_group": run_group,
                    "config_path": f"configs/evolution/{run_group}/{base_id}.yaml",
                    "results_dir": f"artifacts/wfa/runs/{run_group}/{kind}_{base_id}",
                    "status": "completed",
                    "metrics_present": "true",
                    "sharpe_ratio_abs": sharpe,
                    "max_drawdown_on_equity": dd,
                    "total_trades": "240",
                    "total_pairs_traded": "30",
                    "wf_zero_pair_steps_pct": "0.01",
                    "tail_loss_worst_pair_share": "0.20",
                    "tail_loss_worst_period_share": "0.25",
                }
            )


def test_build_factor_pool_joins_decisions(tmp_path: Path) -> None:
    module = _load_script(tmp_path.name)
    run_index = tmp_path / "run_index.csv"
    candidate_id = "evo_abcdef123456"
    _write_run_index(run_index, run_group="rgpool", candidate_id=candidate_id)

    decisions_dir = tmp_path / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    decision = {
        "controller_group": "ctrl_pool",
        "proposals": [
            {
                "candidate_id": candidate_id,
                "parents": ["parent_x"],
                "patch_path": "configs/evolution/rgpool/rgpool_v001_evo_abcdef123456.patch.yaml",
                "patch_ir": {
                    "hypothesis": {"thesis": "Тестовая гипотеза про улучшение risk/selection."},
                    "factors": [
                        {"target_key": "pair_selection.max_pairs", "op": "set", "value": 32, "rationale": "ok"},
                    ],
                },
            }
        ],
    }
    (decisions_dir / "d.json").write_text(json.dumps(decision, ensure_ascii=False) + "\n", encoding="utf-8")

    out_json = tmp_path / "pool.json"
    out_md = tmp_path / "pool.md"
    rc = module.main(
        [
            "--controller-group",
            "ctrl_pool",
            "--run-index",
            str(run_index),
            "--decisions-dir",
            str(decisions_dir),
            "--contains",
            "rgpool",
            "--top",
            "5",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["controller_group"] == "ctrl_pool"
    assert payload["top"][0]["candidate_id"] == candidate_id
    assert "гипотеза" in (payload["top"][0]["hypothesis_thesis"] or "").lower()

