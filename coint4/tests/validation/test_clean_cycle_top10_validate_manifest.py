import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


EXPECTED_WALK_FORWARD = {
    "start_date": "2024-05-01",
    "end_date": "2024-12-31",
    "training_period_days": 90,
    "testing_period_days": 30,
    "step_size_days": 30,
    "max_steps": 5,
    "gap_minutes": 15,
    "refit_frequency": "weekly",
}


def _config_yaml(*, run_id: str, walk_forward: dict) -> str:
    wf = walk_forward
    return (
        "meta:\n"
        f"  id: {run_id}\n"
        "walk_forward:\n"
        f"  start_date: '{wf['start_date']}'\n"
        f"  end_date: '{wf['end_date']}'\n"
        f"  training_period_days: {wf['training_period_days']}\n"
        f"  testing_period_days: {wf['testing_period_days']}\n"
        f"  step_size_days: {wf['step_size_days']}\n"
        f"  max_steps: {wf['max_steps']}\n"
        f"  gap_minutes: {wf['gap_minutes']}\n"
        f"  refit_frequency: {wf['refit_frequency']}\n"
    )


def _make_manifest(tmp_path: Path, *, walk_forward: dict) -> Path:
    runs_root = tmp_path / "runs"
    cfg_root = tmp_path / "configs"
    runs_root.mkdir(parents=True, exist_ok=True)
    cfg_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i in range(10):
        run_id = f"run_{i:02d}"
        run_group = "g0"

        results_dir = runs_root / run_id
        results_dir.mkdir(parents=True, exist_ok=True)
        _write_text(results_dir / "equity_curve.csv", "equity\n100\n101\n")
        _write_text(results_dir / "strategy_metrics.csv", "sharpe_ratio_abs,total_pnl,max_drawdown_abs\n1.0,1.0,-1.0\n")

        cfg_path = cfg_root / f"{run_id}.yaml"
        _write_text(cfg_path, _config_yaml(run_id=run_id, walk_forward=walk_forward))

        manifest.append(
            {
                "run_group": run_group,
                "run_id": run_id,
                "results_dir": str(results_dir),
                "config_path": str(cfg_path),
                "status": "completed",
                "rank": i + 1,
                "rank_sharpe": 1.0,
                "rank_pnl_abs": 1.0,
                "rank_max_drawdown_abs": -1.0,
                "total_trades": 1000,
                "total_costs": 12.34,
                "equity_present": True,
                "metrics_present": True,
                "config_sha256": _sha256_file(cfg_path),
            }
        )

    out = tmp_path / "baseline_manifest.json"
    _write_text(out, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def test_validate_manifest_strict_ok(tmp_path: Path) -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts/optimization/clean_cycle_top10/validate_manifest.py"
    assert script.exists()

    manifest = _make_manifest(tmp_path, walk_forward=EXPECTED_WALK_FORWARD)
    fixed = tmp_path / "fixed_windows.json"
    _write_text(fixed, json.dumps({"walk_forward": EXPECTED_WALK_FORWARD}) + "\n")

    cmd = [
        sys.executable,
        str(script),
        "--manifest",
        str(manifest),
        "--fixed-windows-json",
        str(fixed),
        "--strict",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def test_validate_manifest_strict_fails_on_mismatch(tmp_path: Path) -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts/optimization/clean_cycle_top10/validate_manifest.py"
    assert script.exists()

    bad = dict(EXPECTED_WALK_FORWARD)
    bad["testing_period_days"] = 31
    manifest = _make_manifest(tmp_path, walk_forward=bad)
    fixed = tmp_path / "fixed_windows.json"
    _write_text(fixed, json.dumps({"walk_forward": EXPECTED_WALK_FORWARD}) + "\n")

    cmd = [
        sys.executable,
        str(script),
        "--manifest",
        str(manifest),
        "--fixed-windows-json",
        str(fixed),
        "--strict",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "testing_period_days" in proc.stdout


def test_validate_manifest_report_only_does_not_fail(tmp_path: Path) -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts/optimization/clean_cycle_top10/validate_manifest.py"
    assert script.exists()

    bad = dict(EXPECTED_WALK_FORWARD)
    bad["step_size_days"] = 29
    manifest = _make_manifest(tmp_path, walk_forward=bad)

    cmd = [
        sys.executable,
        str(script),
        "--manifest",
        str(manifest),
        "--use-definitions",
        "--report-only",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "walk_forward mismatches" in proc.stdout

