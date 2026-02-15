import json
import subprocess
import sys
from pathlib import Path


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_compare_metrics_reads_strategy_metrics_and_canonical(tmp_path: Path) -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts/optimization/clean_cycle_top10/compare_metrics.py"
    assert script.exists()

    runs_root = tmp_path / "runs"
    run_ok = runs_root / "run_ok"
    run_diff = runs_root / "run_diff"
    run_ok.mkdir(parents=True, exist_ok=True)
    run_diff.mkdir(parents=True, exist_ok=True)

    _write_text(run_ok / "strategy_metrics.csv", "sharpe_ratio_abs,total_pnl,max_drawdown_abs\n1.0,10.0,-5.0\n")
    _write_text(
        run_ok / "canonical_metrics.json",
        json.dumps(
            {
                "metrics": {
                    "canonical_sharpe": 1.0,
                    "canonical_pnl_abs": 10.0,
                    "canonical_max_drawdown_abs": -5.0,
                }
            }
        )
        + "\n",
    )

    _write_text(run_diff / "strategy_metrics.csv", "sharpe_ratio_abs,total_pnl,max_drawdown_abs\n1.0,10.0,-5.0\n")
    _write_text(
        run_diff / "canonical_metrics.json",
        json.dumps(
            {
                "metrics": {
                    "canonical_sharpe": 2.0,
                    "canonical_pnl_abs": 10.0,
                    "canonical_max_drawdown_abs": -5.0,
                }
            }
        )
        + "\n",
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"results_dir": str(run_ok)},
                {"results_dir": str(run_diff)},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(script),
        "--manifest",
        str(manifest),
        "--thresholds",
        "sharpe=2,pnl=0,dd=0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0

    # Ensure the diff is computed as Δ = canonical - raw (2.0 - 1.0 = +1.0).
    assert "|  | run_diff |" in proc.stdout
    assert "+1.000000" in proc.stdout
    assert "- selected: 2" in proc.stdout


def test_compare_metrics_sampling_top_and_random_is_deterministic(tmp_path: Path) -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts/optimization/clean_cycle_top10/compare_metrics.py"
    assert script.exists()

    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    def _mk(run_id: str, sharpe: float) -> Path:
        d = runs_root / run_id
        d.mkdir(parents=True, exist_ok=True)
        _write_text(d / "strategy_metrics.csv", f"sharpe_ratio_abs,total_pnl,max_drawdown_abs\n{sharpe},0.0,0.0\n")
        _write_text(
            d / "canonical_metrics.json",
            json.dumps(
                {
                    "metrics": {
                        "canonical_sharpe": sharpe,
                        "canonical_pnl_abs": 0.0,
                        "canonical_max_drawdown_abs": 0.0,
                    }
                }
            )
            + "\n",
        )
        return d

    a = _mk("run_a", 4.0)
    b = _mk("run_b", 3.0)
    c = _mk("run_c", 2.0)
    _ = _mk("run_d", 1.0)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"results_dir": str(a)},
                {"results_dir": str(b)},
                {"results_dir": str(c)},
                {"results_dir": str(runs_root / "run_d")},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(script),
        "--manifest",
        str(manifest),
        "--sample",
        "top2+random1",
        "--seed",
        "1",
        "--thresholds",
        "sharpe=0,pnl=0,dd=0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0

    # With seed=1 and remaining=[run_c, run_d], random1 deterministically picks run_c.
    assert "|  | run_a |" in proc.stdout
    assert "|  | run_b |" in proc.stdout
    assert "|  | run_c |" in proc.stdout
    assert "run_d" not in proc.stdout


def test_compare_metrics_dry_run_does_not_require_run_files(tmp_path: Path) -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts/optimization/clean_cycle_top10/compare_metrics.py"
    assert script.exists()

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("results_dir,sharpe_ratio_abs_raw,total_pnl,max_drawdown_abs\n/does/not/exist,1,2,3\n", encoding="utf-8")

    cmd = [
        sys.executable,
        str(script),
        "--manifest",
        str(manifest),
        "--dry-run",
        "--sample",
        "top1+random1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "dry-run" in proc.stdout.lower()
    assert "targets" in proc.stdout.lower()

