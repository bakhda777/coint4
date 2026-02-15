import csv
import json
import subprocess
import sys
from pathlib import Path


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(r) for r in reader]


def test_build_clean_rollup_is_deterministic_and_filtered(tmp_path: Path) -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts/optimization/clean_cycle_top10/build_clean_rollup.py"
    assert script.exists()

    runs_root = tmp_path / "runs"
    cfg_root = tmp_path / "configs"
    runs_root.mkdir(parents=True, exist_ok=True)
    cfg_root.mkdir(parents=True, exist_ok=True)

    # Completed + canonical: should be included by default.
    run_ok_1 = runs_root / "run_ok_01"
    run_ok_1.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_ok_1 / "canonical_metrics.json",
        {
            "schema_version": 1,
            "metrics": {
                "canonical_sharpe": 1.0,
                "canonical_pnl_abs": 10.0,
                "canonical_max_drawdown_abs": -1.0,
            },
        },
    )
    cfg_ok_1 = cfg_root / "b01.yaml"
    _write_text(cfg_ok_1, "meta: {id: b01}\n")

    run_ok_2 = runs_root / "run_ok_02"
    run_ok_2.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_ok_2 / "canonical_metrics.json",
        {
            "schema_version": 1,
            "metrics": {
                "canonical_sharpe": 1.1,
                "canonical_pnl_abs": 20.0,
                "canonical_max_drawdown_abs": -10.0,
            },
        },
    )
    cfg_ok_2 = cfg_root / "b02.yaml"
    _write_text(cfg_ok_2, "meta: {id: b02}\n")

    # Planned (non-completed) but canonical: should be excluded by default.
    run_planned = runs_root / "run_planned"
    run_planned.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_planned / "canonical_metrics.json",
        {
            "metrics": {
                "canonical_sharpe": 99.0,
                "canonical_pnl_abs": 0.0,
                "canonical_max_drawdown_abs": -1.0,
            }
        },
    )
    cfg_planned = cfg_root / "planned.yaml"
    _write_text(cfg_planned, "meta: {id: planned}\n")

    # Completed but missing canonical_metrics.json: excluded by default.
    run_missing_canon = runs_root / "run_missing_canon"
    run_missing_canon.mkdir(parents=True, exist_ok=True)
    cfg_missing = cfg_root / "missing.yaml"
    _write_text(cfg_missing, "meta: {id: missing}\n")

    baseline_manifest = tmp_path / "baseline_manifest.json"
    _write_json(
        baseline_manifest,
        [
            {
                "rank": 1,
                "run_group": "g0",
                "run_id": "old_run_01",
                "status": "completed",
                "results_dir": str(run_ok_1),
                "config_path": str(cfg_ok_1),
                "config_sha256": "",
            },
            {
                "rank": 2,
                "run_group": "g0",
                "run_id": "old_run_02",
                "status": "completed",
                "results_dir": str(run_ok_2),
                "config_path": str(cfg_ok_2),
                "config_sha256": "",
            },
            {
                "rank": 3,
                "run_group": "g0",
                "run_id": "old_run_planned",
                "status": "planned",
                "results_dir": str(run_planned),
                "config_path": str(cfg_planned),
                "config_sha256": "",
            },
            {
                "rank": 4,
                "run_group": "g0",
                "run_id": "old_run_missing_canon",
                "status": "completed",
                "results_dir": str(run_missing_canon),
                "config_path": str(cfg_missing),
                "config_sha256": "",
            },
        ],
    )

    sweeps_manifest = tmp_path / "sweeps_manifest.json"
    run_sweep = runs_root / "run_sweep_alpha"
    run_sweep.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_sweep / "canonical_metrics.json",
        {
            "metrics": {
                "canonical_sharpe": 2.0,
                "canonical_pnl_abs": 1.0,
                "canonical_max_drawdown_abs": -100.0,
            }
        },
    )
    cfg_sweep = cfg_root / "sweep_alpha.yaml"
    _write_text(cfg_sweep, "meta: {id: sweep_alpha}\n")
    _write_json(
        sweeps_manifest,
        [
            {
                "status": "completed",
                "results_dir": str(run_sweep),
                "config_path": str(cfg_sweep),
            }
        ],
    )

    out_csv = tmp_path / "rollup.csv"
    out_md = tmp_path / "rollup.md"
    cmd = [
        sys.executable,
        str(script),
        "--baseline-manifest",
        str(baseline_manifest),
        "--sweeps-manifest",
        str(sweeps_manifest),
        "--output-csv",
        str(out_csv),
        "--output-md",
        str(out_md),
        "--overwrite",
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    csv_first = out_csv.read_bytes()
    md_first = out_md.read_bytes()
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    csv_second = out_csv.read_bytes()
    md_second = out_md.read_bytes()

    assert csv_first == csv_second
    assert md_first == md_second

    rows = _read_csv_rows(out_csv)
    assert len(rows) == 3  # completed + canonical only

    # Sorting: score = sharpe - 0.02 * abs(dd)
    # run_ok_01: 1.0 - 0.02*1 = 0.98
    # run_ok_02: 1.1 - 0.02*10 = 0.90
    # sweep_alpha: 2.0 - 0.02*100 = 0.00
    assert rows[0]["run_name"] == "b01"
    assert rows[1]["run_name"] == "b02"
    assert rows[2]["run_name"] == "sweep_alpha"

    # config_sha256 should be computed when missing (non-empty hex).
    assert len(rows[0]["config_sha256"]) == 64
    assert all(c in "0123456789abcdef" for c in rows[0]["config_sha256"])

    # Fixed windows fingerprint should be present (sha256 hex).
    assert len(rows[0]["fixed_windows_fingerprint"]) == 64
    assert all(c in "0123456789abcdef" for c in rows[0]["fixed_windows_fingerprint"])


def test_build_clean_rollup_can_include_noncompleted_and_missing_canonical(tmp_path: Path) -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts/optimization/clean_cycle_top10/build_clean_rollup.py"
    assert script.exists()

    runs_root = tmp_path / "runs"
    cfg_root = tmp_path / "configs"
    runs_root.mkdir(parents=True, exist_ok=True)
    cfg_root.mkdir(parents=True, exist_ok=True)

    run_ok = runs_root / "ok"
    run_ok.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_ok / "canonical_metrics.json",
        {
            "metrics": {
                "canonical_sharpe": 1.0,
                "canonical_pnl_abs": 1.0,
                "canonical_max_drawdown_abs": -1.0,
            }
        },
    )
    cfg_ok = cfg_root / "ok.yaml"
    _write_text(cfg_ok, "meta: {id: ok}\n")

    run_planned = runs_root / "planned"
    run_planned.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_planned / "canonical_metrics.json",
        {
            "metrics": {
                "canonical_sharpe": 2.0,
                "canonical_pnl_abs": 2.0,
                "canonical_max_drawdown_abs": -2.0,
            }
        },
    )
    cfg_planned = cfg_root / "planned.yaml"
    _write_text(cfg_planned, "meta: {id: planned}\n")

    run_missing = runs_root / "missing"
    run_missing.mkdir(parents=True, exist_ok=True)
    cfg_missing = cfg_root / "missing.yaml"
    _write_text(cfg_missing, "meta: {id: missing}\n")

    baseline_manifest = tmp_path / "baseline_manifest.json"
    _write_json(
        baseline_manifest,
        [
            {"status": "completed", "results_dir": str(run_ok), "config_path": str(cfg_ok)},
            {"status": "planned", "results_dir": str(run_planned), "config_path": str(cfg_planned)},
            {"status": "completed", "results_dir": str(run_missing), "config_path": str(cfg_missing)},
        ],
    )

    out_csv = tmp_path / "rollup.csv"
    out_md = tmp_path / "rollup.md"
    cmd = [
        sys.executable,
        str(script),
        "--baseline-manifest",
        str(baseline_manifest),
        "--output-csv",
        str(out_csv),
        "--output-md",
        str(out_md),
        "--include-noncompleted",
        "--include-missing-canonical",
        "--overwrite",
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    rows = _read_csv_rows(out_csv)
    assert len(rows) == 3
    assert {r["status"] for r in rows} == {"completed", "planned"}
    assert any(r["canonical_metrics_present"] == "false" for r in rows)

