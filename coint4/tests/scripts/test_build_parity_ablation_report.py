from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_script(tmp_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts/optimization/build_parity_ablation_report.py"
    spec = importlib.util.spec_from_file_location(f"build_parity_ablation_report_test_{tmp_name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_parity_ablation_report_outputs_files(tmp_path: Path) -> None:
    module = _load_script(tmp_path.name)

    run_index = tmp_path / "run_index.csv"
    with run_index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_group",
                "score",
                "worst_robust_sharpe",
                "worst_dd_pct",
                "sharpe_ratio_abs",
                "max_drawdown_on_equity",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_group": "rg1",
                "score": "1.2",
                "worst_robust_sharpe": "1.1",
                "worst_dd_pct": "0.12",
                "sharpe_ratio_abs": "1.1",
                "max_drawdown_on_equity": "-0.12",
            }
        )

    decisions_dir = tmp_path / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    (decisions_dir / "d1.json").write_text(
        json.dumps(
            {
                "controller_group": "ctrl1",
                "run_group": "rg1",
                "rng": {"algorithm": "PCG64", "state": {"v": 1}},
                "llm_policy": {"used": True},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    rc = module.main(
        [
            "--controller-group",
            "ctrl1",
            "--run-index",
            str(run_index),
            "--decisions-dir",
            str(decisions_dir),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )
    assert rc == 0
    assert output_json.exists()
    assert output_md.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["controller_group"] == "ctrl1"
    assert payload["decisions_total"] == 1
