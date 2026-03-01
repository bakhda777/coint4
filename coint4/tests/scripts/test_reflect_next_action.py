from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path


def _load_script(tmp_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts/optimization/reflect_next_action.py"
    spec = importlib.util.spec_from_file_location(f"reflect_next_action_test_{tmp_name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_run_index(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "run_group",
                "config_path",
                "results_dir",
                "status",
                "metrics_present",
                "sharpe_ratio_abs",
                "max_drawdown_on_equity",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_reflect_next_action_generates_json(tmp_path: Path) -> None:
    module = _load_script(tmp_path.name)
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "decision_id": "d1",
                "run_group": "rg_reflect",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    run_index = tmp_path / "run_index.csv"
    _write_run_index(
        run_index,
        rows=[
            {
                "run_id": "holdout_rg_reflect_v1_oos20220101_20221231",
                "run_group": "rg_reflect",
                "config_path": "cfg.yaml",
                "results_dir": "res",
                "status": "completed",
                "metrics_present": "true",
                "sharpe_ratio_abs": "1.1",
                "max_drawdown_on_equity": "-0.10",
            }
        ],
    )
    output_path = tmp_path / "reflection.json"
    rc = module.main(
        [
            "--decision",
            str(decision_path),
            "--run-index",
            str(run_index),
            "--output-json",
            str(output_path),
            "--contains",
            "rg_reflect",
        ]
    )
    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["critic_source"] == "deterministic"
    assert payload["next_action"] in {"run_next_batch", "wait", "stop"}


def test_reflect_next_action_single_bad_window_keeps_running(tmp_path: Path) -> None:
    module = _load_script(f"{tmp_path.name}_single_bad")
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps({"decision_id": "d2", "run_group": "rg_one_bad"}), encoding="utf-8")
    run_index = tmp_path / "run_index.csv"
    rows = []
    for idx in range(5):
        rows.append(
            {
                "run_id": f"good_{idx}",
                "run_group": "rg_one_bad",
                "config_path": "cfg.yaml",
                "results_dir": f"res_good_{idx}",
                "status": "completed",
                "metrics_present": "true",
                "sharpe_ratio_abs": "1.2",
                "max_drawdown_on_equity": "-0.10",
            }
        )
    rows.append(
        {
            "run_id": "bad_0",
            "run_group": "rg_one_bad",
            "config_path": "cfg_bad.yaml",
            "results_dir": "res_bad",
            "status": "completed",
            "metrics_present": "true",
            "sharpe_ratio_abs": "-3.9",
            "max_drawdown_on_equity": "-0.26",
        }
    )
    _write_run_index(run_index, rows=rows)

    output_path = tmp_path / "reflection.json"
    rc = module.main(
        [
            "--decision",
            str(decision_path),
            "--run-index",
            str(run_index),
            "--output-json",
            str(output_path),
            "--contains",
            "rg_one_bad",
        ]
    )
    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["next_action"] == "run_next_batch"
    assert payload["result"]["bad_rows_count"] == 1


def test_reflect_next_action_systemic_bad_batch_stops(tmp_path: Path) -> None:
    module = _load_script(f"{tmp_path.name}_systemic_bad")
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps({"decision_id": "d3", "run_group": "rg_systemic_bad"}), encoding="utf-8")
    run_index = tmp_path / "run_index.csv"
    _write_run_index(
        run_index,
        rows=[
            {
                "run_id": "bad_1",
                "run_group": "rg_systemic_bad",
                "config_path": "cfg_bad1.yaml",
                "results_dir": "res_bad1",
                "status": "completed",
                "metrics_present": "true",
                "sharpe_ratio_abs": "-1.0",
                "max_drawdown_on_equity": "-0.30",
            },
            {
                "run_id": "bad_2",
                "run_group": "rg_systemic_bad",
                "config_path": "cfg_bad2.yaml",
                "results_dir": "res_bad2",
                "status": "completed",
                "metrics_present": "true",
                "sharpe_ratio_abs": "-0.4",
                "max_drawdown_on_equity": "-0.28",
            },
            {
                "run_id": "good_1",
                "run_group": "rg_systemic_bad",
                "config_path": "cfg_good1.yaml",
                "results_dir": "res_good1",
                "status": "completed",
                "metrics_present": "true",
                "sharpe_ratio_abs": "0.9",
                "max_drawdown_on_equity": "-0.12",
            },
            {
                "run_id": "good_2",
                "run_group": "rg_systemic_bad",
                "config_path": "cfg_good2.yaml",
                "results_dir": "res_good2",
                "status": "completed",
                "metrics_present": "true",
                "sharpe_ratio_abs": "0.8",
                "max_drawdown_on_equity": "-0.11",
            },
        ],
    )

    output_path = tmp_path / "reflection.json"
    rc = module.main(
        [
            "--decision",
            str(decision_path),
            "--run-index",
            str(run_index),
            "--output-json",
            str(output_path),
            "--contains",
            "rg_systemic_bad",
        ]
    )
    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    # Default: stop is suppressed; bad batches should keep iterating.
    assert payload["next_action"] == "run_next_batch"
    assert payload["result"]["bad_rows_count"] == 2


def test_reflect_next_action_hard_stop_on_extreme_row(tmp_path: Path) -> None:
    module = _load_script(f"{tmp_path.name}_hard_stop")
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps({"decision_id": "d4", "run_group": "rg_hard_stop"}), encoding="utf-8")
    run_index = tmp_path / "run_index.csv"
    _write_run_index(
        run_index,
        rows=[
            {
                "run_id": "extreme_bad",
                "run_group": "rg_hard_stop",
                "config_path": "cfg_extreme.yaml",
                "results_dir": "res_extreme",
                "status": "completed",
                "metrics_present": "true",
                "sharpe_ratio_abs": "-2.5",
                "max_drawdown_on_equity": "-0.41",
            }
        ],
    )

    output_path = tmp_path / "reflection.json"
    rc = module.main(
        [
            "--decision",
            str(decision_path),
            "--run-index",
            str(run_index),
            "--output-json",
            str(output_path),
            "--contains",
            "rg_hard_stop",
        ]
    )
    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    # Default: stop is suppressed, but the risk flag must remain.
    assert payload["next_action"] == "run_next_batch"
    assert "hard_stop_triggered" in payload["risk_flags"]
    assert "stop_suppressed" in payload["risk_flags"]


def test_reflect_next_action_stop_can_be_enabled_via_env(tmp_path: Path) -> None:
    module = _load_script(f"{tmp_path.name}_stop_enabled")
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps({"decision_id": "d5", "run_group": "rg_stop_enabled"}), encoding="utf-8")
    run_index = tmp_path / "run_index.csv"
    _write_run_index(
        run_index,
        rows=[
            {
                "run_id": "extreme_bad",
                "run_group": "rg_stop_enabled",
                "config_path": "cfg_extreme.yaml",
                "results_dir": "res_extreme",
                "status": "completed",
                "metrics_present": "true",
                "sharpe_ratio_abs": "-2.5",
                "max_drawdown_on_equity": "-0.41",
            }
        ],
    )

    output_path = tmp_path / "reflection.json"
    prev = os.environ.get("COINT4_REFLECT_ENABLE_STOP")
    os.environ["COINT4_REFLECT_ENABLE_STOP"] = "1"
    try:
        rc = module.main(
            [
                "--decision",
                str(decision_path),
                "--run-index",
                str(run_index),
                "--output-json",
                str(output_path),
                "--contains",
                "rg_stop_enabled",
            ]
        )
    finally:
        if prev is None:
            os.environ.pop("COINT4_REFLECT_ENABLE_STOP", None)
        else:
            os.environ["COINT4_REFLECT_ENABLE_STOP"] = prev
    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["next_action"] == "stop"
