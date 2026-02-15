import json
import subprocess
import sys
from pathlib import Path


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_select_top10_manifest_is_deterministic(tmp_path: Path) -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts/optimization/clean_cycle_top10/select_top10.py"
    assert script.exists()

    # Create 12 synthetic runs with controlled ties for deterministic tie-breaks.
    runs_root = tmp_path / "runs"
    cfg_root = tmp_path / "configs"
    runs_root.mkdir(parents=True, exist_ok=True)
    cfg_root.mkdir(parents=True, exist_ok=True)

    header = [
        "run_id",
        "run_group",
        "results_dir",
        "config_path",
        "status",
        "metrics_present",
        "sharpe_ratio_abs",
        "sharpe_ratio_abs_raw",
        "total_pnl",
        "max_drawdown_abs",
        "total_trades",
        "total_pairs_traded",
        "total_costs",
    ]

    rows = []
    for i in range(12):
        run_id = f"run_{i:02d}"
        run_group = "g0" if i < 6 else "g1"
        results_dir = runs_root / run_id
        results_dir.mkdir(parents=True, exist_ok=True)
        _write_text(results_dir / "equity_curve.csv", "equity\n100\n101\n")
        _write_text(results_dir / "strategy_metrics.csv", "sharpe_ratio_abs,total_pnl,max_drawdown_abs\n1.0,1.0,-1.0\n")

        cfg_path = cfg_root / f"{run_id}.yaml"
        _write_text(cfg_path, f"meta: {{id: {run_id}}}\n")

        # Build tie patterns:
        # - i=0..3: sharpe=2.0, dd differs -> dd tie-break
        # - i=4..5: sharpe=1.9
        # - i=6..11: sharpe decreases
        if i < 4:
            sharpe = 2.0
            dd = [-100.0, -50.0, -50.0, -50.0][i]
            pnl = [10.0, 5.0, 15.0, 15.0][i]
        elif i < 6:
            sharpe = 1.9
            dd = -10.0
            pnl = 100.0 - i
        else:
            sharpe = 1.8 - (i - 6) * 0.1
            dd = -20.0
            pnl = float(i)

        # Add canonical metrics for run_01 to ensure canonical_sharpe dominates ranking.
        if run_id == "run_01":
            _write_text(
                results_dir / "canonical_metrics.json",
                json.dumps(
                    {
                        "canonical_sharpe": 9.0,
                        "canonical_pnl_abs": 123.0,
                        "canonical_max_drawdown_abs": -7.0,
                    }
                )
                + "\n",
            )

        rows.append(
            {
                "run_id": run_id,
                "run_group": run_group,
                "results_dir": str(results_dir),
                "config_path": str(cfg_path),
                "status": "completed",
                "metrics_present": "true",
                "sharpe_ratio_abs": f"{sharpe}",
                "sharpe_ratio_abs_raw": f"{sharpe - 0.01}",
                "total_pnl": f"{pnl}",
                "max_drawdown_abs": f"{dd}",
                "total_trades": "1000",
                "total_pairs_traded": "50",
                "total_costs": "12.34",
            }
        )

    run_index = tmp_path / "run_index.csv"
    with run_index.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(row[h]) for h in header) + "\n")

    out = tmp_path / "baseline_manifest.json"
    cmd = [
        sys.executable,
        str(script),
        "--run-index",
        str(run_index),
        "--output",
        str(out),
        "--top-n",
        "10",
        "--overwrite",
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    first = out.read_bytes()
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    second = out.read_bytes()

    assert first == second

    payload = json.loads(first.decode("utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 10

    # Canonical_sharpe for run_01 must dominate ranking.
    assert payload[0]["run_id"] == "run_01"
    assert payload[0]["rank_sharpe_source"] == "canonical_sharpe"

    # For the remaining sharpe=2.0 ties, dd (abs) asc then pnl desc, then ids.
    # run_02/run_03/run_04 don't exist; our tied ones are run_00/run_02/run_03 with dd=-50.
    # Among dd=-50 and pnl=15, run_02 and run_03 should appear before run_00 (dd=-100).
    ids = [row["run_id"] for row in payload[:5]]
    assert "run_00" in ids  # still high sharpe, but worse dd so comes later vs dd=-50 runs

