from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_script(tmp_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts/optimization/transfer_generalization_report.py"
    spec = importlib.util.spec_from_file_location(f"transfer_generalization_report_test_{tmp_name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_transfer_generalization_report_builds_outputs(tmp_path: Path) -> None:
    module = _load_script(tmp_path.name)
    run_index = tmp_path / "run_index.csv"
    with run_index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "run_group",
                "config_path",
                "status",
                "metrics_present",
                "sharpe_ratio_abs",
                "max_drawdown_on_equity",
                "total_pnl",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "holdout_rg_v1_oos20220101_20221231",
                "run_group": "rg",
                "config_path": "cfg1.yaml",
                "status": "completed",
                "metrics_present": "true",
                "sharpe_ratio_abs": "1.5",
                "max_drawdown_on_equity": "-0.10",
                "total_pnl": "120",
            }
        )
    output_json = tmp_path / "transfer.json"
    output_md = tmp_path / "transfer.md"
    rc = module.main(
        [
            "--run-index",
            str(run_index),
            "--contains",
            "rg",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )
    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["variants_count"] == 1
    assert output_md.exists()
