from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


def _load_script(tmp_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts/optimization/evolve_next_batch.py"
    spec = importlib.util.spec_from_file_location(f"evolve_next_batch_stress_naming_{tmp_name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_run_index(path: Path, *, cfg_path: Path, run_group: str) -> None:
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
    base_id = f"{run_group}_seed_oos20220101_20221231"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "run_id": f"holdout_{base_id}",
                "run_group": run_group,
                "config_path": str(cfg_path),
                "results_dir": f"artifacts/wfa/runs/{run_group}/holdout_{base_id}",
                "status": "completed",
                "metrics_present": "true",
                "sharpe_ratio_abs": "1.4",
                "max_drawdown_on_equity": "-0.08",
                "total_trades": "260",
                "total_pairs_traded": "32",
                "wf_zero_pair_steps_pct": "0.01",
                "tail_loss_worst_pair_share": "0.20",
                "tail_loss_worst_period_share": "0.25",
            }
        )


def test_evolve_next_batch_uses_suffix_stress_config_naming(tmp_path: Path) -> None:
    module = _load_script(tmp_path.name)
    base_cfg = tmp_path / "base.yaml"
    base_cfg.write_text(
        "\n".join(
            [
                "walk_forward:",
                "  start_date: '2022-01-01'",
                "  end_date: '2022-12-31'",
                "portfolio:",
                "  risk_per_position_pct: 0.01",
                "  max_active_positions: 16",
                "backtest:",
                "  zscore_entry_threshold: 1.2",
                "  zscore_exit: 0.15",
                "  rolling_window: 96",
                "pair_selection:",
                "  max_pairs: 24",
                "  min_correlation: 0.4",
                "  coint_pvalue_threshold: 0.2",
                "filter_params:",
                "  max_hurst_exponent: 0.8",
                "  min_mean_crossings: 2",
                "  max_half_life_days: 60",
                "",
            ]
        ),
        encoding="utf-8",
    )
    run_index = tmp_path / "run_index.csv"
    _write_run_index(run_index, cfg_path=base_cfg, run_group="rgnaming")

    queue_root = tmp_path / "queue"
    configs_root = tmp_path / "configs"
    runs_root = tmp_path / "runs"

    rc = module.main(
        [
            "--base-config",
            str(base_cfg),
            "--controller-group",
            "ctrl_naming",
            "--run-group",
            "rgnaming_next",
            "--run-index",
            str(run_index),
            "--contains",
            "rgnaming",
            "--num-variants",
            "1",
            "--dedupe-distance",
            "0.0",
            "--min-windows",
            "1",
            "--window",
            "2022-01-01,2022-12-31",
            "--queue-dir",
            str(queue_root),
            "--configs-dir",
            str(configs_root),
            "--runs-dir",
            str(runs_root),
            "--include-stress",
        ]
    )
    assert rc == 0

    queue_path = queue_root / "rgnaming_next" / "run_queue.csv"
    with queue_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    stress_rows = [row for row in rows if Path(str(row["results_dir"])).name.startswith("stress_")]
    assert len(stress_rows) == 1

    stress_cfg_name = Path(str(stress_rows[0]["config_path"])).name
    assert stress_cfg_name.endswith("_stress.yaml")
    assert not stress_cfg_name.startswith("stress_")

    stress_cfg_path = configs_root / "rgnaming_next" / stress_cfg_name
    assert stress_cfg_path.exists()

    variant_stem = stress_cfg_name.removesuffix("_stress.yaml")
    legacy_stress_path = configs_root / "rgnaming_next" / f"stress_{variant_stem}.yaml"
    assert not legacy_stress_path.exists()
