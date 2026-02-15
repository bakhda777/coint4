import csv
import json
import subprocess
import sys
from pathlib import Path


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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _config_yaml(*, walk_forward: dict) -> str:
    wf = walk_forward
    return (
        "meta:\n"
        "  id: sweep\n"
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


def _read_queue(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def test_build_sweeps_queue_from_configs_dir_ok(tmp_path: Path) -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts/optimization/clean_cycle_top10/build_sweeps_queue.py"
    assert script.exists()

    cfg_dir = tmp_path / "sweeps"
    cfg_a = cfg_dir / "s001_alpha.yaml"
    cfg_b = cfg_dir / "s002_beta.yaml"
    _write_text(cfg_a, _config_yaml(walk_forward=EXPECTED_WALK_FORWARD))
    _write_text(cfg_b, _config_yaml(walk_forward=EXPECTED_WALK_FORWARD))

    out = tmp_path / "sweeps_run_queue.csv"
    cmd = [
        sys.executable,
        str(script),
        "--configs-dir",
        str(cfg_dir),
        "--opt-dir",
        "opt_sweeps",
        "--output",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    rows = _read_queue(out)
    assert [Path(r["config_path"]).name for r in rows] == ["s001_alpha.yaml", "s002_beta.yaml"]
    assert [r["results_dir"] for r in rows] == ["opt_sweeps/s001_alpha", "opt_sweeps/s002_beta"]
    assert [r["status"] for r in rows] == ["planned", "planned"]


def test_build_sweeps_queue_from_manifest_ok(tmp_path: Path) -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts/optimization/clean_cycle_top10/build_sweeps_queue.py"
    assert script.exists()

    cfg_dir = tmp_path / "sweeps"
    cfg_a = cfg_dir / "s001_alpha.yaml"
    cfg_b = cfg_dir / "s002_beta.yaml"
    _write_text(cfg_a, _config_yaml(walk_forward=EXPECTED_WALK_FORWARD))
    _write_text(cfg_b, _config_yaml(walk_forward=EXPECTED_WALK_FORWARD))

    manifest = tmp_path / "sweeps_manifest.json"
    _write_text(
        manifest,
        json.dumps(
            [
                {"config_path": str(cfg_b)},
                {"config_path": str(cfg_a)},
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    out = tmp_path / "sweeps_run_queue.csv"
    cmd = [
        sys.executable,
        str(script),
        "--sweeps-manifest",
        str(manifest),
        "--opt-dir",
        "opt_sweeps",
        "--output",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    rows = _read_queue(out)
    # Manifest order should be preserved.
    assert [Path(r["config_path"]).name for r in rows] == ["s002_beta.yaml", "s001_alpha.yaml"]
    assert [r["results_dir"] for r in rows] == ["opt_sweeps/s002_beta", "opt_sweeps/s001_alpha"]


def test_build_sweeps_queue_fails_on_unsafe_max_steps(tmp_path: Path) -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts/optimization/clean_cycle_top10/build_sweeps_queue.py"
    assert script.exists()

    bad_wf = dict(EXPECTED_WALK_FORWARD)
    bad_wf["max_steps"] = 6

    cfg_dir = tmp_path / "sweeps"
    cfg_a = cfg_dir / "s001_bad.yaml"
    _write_text(cfg_a, _config_yaml(walk_forward=bad_wf))

    out = tmp_path / "sweeps_run_queue.csv"
    cmd = [
        sys.executable,
        str(script),
        "--configs-dir",
        str(cfg_dir),
        "--opt-dir",
        "opt_sweeps",
        "--output",
        str(out),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "unsafe max_steps" in (proc.stdout + proc.stderr)

