from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path


def _copy_into_fake_repo(*, tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    app_root = repo_root / "coint4"

    (app_root / "scripts" / "optimization").mkdir(parents=True, exist_ok=True)
    (app_root / "configs" / "autopilot").mkdir(parents=True, exist_ok=True)
    (app_root / "configs").mkdir(parents=True, exist_ok=True)
    (app_root / "artifacts" / "wfa" / "aggregate" / "rollup").mkdir(parents=True, exist_ok=True)

    src_app_root = Path(__file__).resolve().parents[2]
    shutil.copy2(
        src_app_root / "scripts" / "optimization" / "autopilot_budget1000.py",
        app_root / "scripts" / "optimization" / "autopilot_budget1000.py",
    )

    # Provide a minimal `coint2.ops.run_queue` without importing the full coint2 package.
    (app_root / "src" / "coint2" / "ops").mkdir(parents=True, exist_ok=True)
    (app_root / "src" / "coint2" / "__init__.py").write_text("", encoding="utf-8")
    (app_root / "src" / "coint2" / "ops" / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(
        src_app_root / "src" / "coint2" / "ops" / "run_queue.py",
        app_root / "src" / "coint2" / "ops" / "run_queue.py",
    )

    return repo_root


def _load_autopilot_module(*, repo_root: Path):
    app_root = repo_root / "coint4"
    module_path = app_root / "scripts" / "optimization" / "autopilot_budget1000.py"
    spec = importlib.util.spec_from_file_location(f"autopilot_budget1000_fake_{repo_root.name}", module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _run_main_isolated(mod, argv: list[str]) -> int:
    before_path = list(sys.path)
    before_modules = set(sys.modules.keys())
    try:
        return int(mod.main(argv))
    finally:
        sys.path[:] = before_path
        # Avoid leaking stub `coint2` into other tests.
        for name in list(sys.modules.keys()):
            if name in before_modules:
                continue
            if name == "coint2" or name.startswith("coint2."):
                sys.modules.pop(name, None)


def _write_queue_completed(queue_path: Path) -> list[dict[str, str]]:
    with queue_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        row["status"] = "completed"
    with queue_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config_path", "results_dir", "status"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _touch_metrics(*, app_root: Path, rows: list[dict[str, str]]) -> None:
    for row in rows:
        results_dir = str(row.get("results_dir") or "").strip()
        assert results_dir
        run_dir = app_root / results_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "strategy_metrics.csv").write_text("ok\n", encoding="utf-8")


def _write_rollup_for_variant(
    *,
    app_root: Path,
    run_group: str,
    queue_rows: list[dict[str, str]],
    risk_tag: str,
) -> None:
    holdout = [r for r in queue_rows if "holdout_" in (r.get("results_dir") or "") and risk_tag in (r.get("results_dir") or "")]
    assert len(holdout) == 1

    def _run_id(results_dir: str) -> str:
        # artifacts/.../<run_group>/<run_id>
        return Path(results_dir).name

    holdout_id = _run_id(str(holdout[0]["results_dir"]))

    rollup_path = app_root / "artifacts" / "wfa" / "aggregate" / "rollup" / "run_index.csv"
    rollup_path.parent.mkdir(parents=True, exist_ok=True)
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
    ]
    with rollup_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "run_id": holdout_id,
                "run_group": run_group,
                "config_path": str(holdout[0]["config_path"]),
                "results_dir": str(holdout[0]["results_dir"]),
                "status": "completed",
                "metrics_present": "True",
                "sharpe_ratio_abs": "2.0",
                "max_drawdown_on_equity": "-0.05",
                "total_trades": "10",
                "total_pairs_traded": "3",
            }
        )


def test_plan_next_advances_completed_round_and_emits_next_queue(tmp_path: Path) -> None:
    repo_root = _copy_into_fake_repo(tmp_path=tmp_path)
    app_root = repo_root / "coint4"
    mod = _load_autopilot_module(repo_root=repo_root)

    # Seed/base config.
    (app_root / "configs" / "seed.yaml").write_text(
        "\n".join(
            [
                "walk_forward:",
                "  start_date: '2022-01-01'",
                "  end_date: '2022-12-31'",
                "portfolio:",
                "  risk_per_position_pct: 0.01",
                "backtest:",
                "  pair_stop_loss_usd: 5.0",
                "  max_var_multiplier: 1.02",
                "",
            ]
        ),
        encoding="utf-8",
    )

    run_group_prefix = "20260226_test"
    controller_group = "20260226_test_controller"
    (app_root / "configs" / "autopilot" / "test.yaml").write_text(
        "\n".join(
            [
                "base_config: configs/seed.yaml",
                "windows:",
                "  - ['2022-01-01','2022-12-31']",
                f"run_group_prefix: '{run_group_prefix}'",
                f"controller_group: '{controller_group}'",
                "configs_dir: 'configs/generated'",
                "queue_dir: 'artifacts/wfa/aggregate'",
                "runs_dir: 'artifacts/wfa/runs_clean'",
                "git:",
                "  auto_stage: false",
                "selection:",
                "  min_windows: 1",
                "  min_trades: 1",
                "  min_pairs: 1",
                "  max_dd_pct: null",
                "  dd_target_pct: null",
                "  dd_penalty: 0.0",
                "  min_psr: null",
                "  min_dsr: null",
                "search:",
                "  max_rounds: 3",
                "  no_improvement_rounds: 2",
                "  min_improvement: 0.02",
                "  knobs:",
                "    - key: 'portfolio.risk_per_position_pct'",
                "      op: add",
                "      step: 0.01",
                "      candidates: [-1, 0, 1]",
                "      min: 0.001",
                "      max: 0.05",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # 1) Plan next: generates r01 queue and exits (not completed).
    rc = _run_main_isolated(mod, ["--config", "configs/autopilot/test.yaml", "--reset", "--plan-next"])
    assert rc == 0

    run_group_r01 = f"{run_group_prefix}_r01_risk"
    queue_r01 = app_root / "artifacts" / "wfa" / "aggregate" / run_group_r01 / "run_queue.csv"
    assert queue_r01.exists()

    # 2) Mark r01 queue as completed + stub metrics; write rollup for the winning variant (risk0p02).
    rows_r01 = _write_queue_completed(queue_r01)
    _touch_metrics(app_root=app_root, rows=rows_r01)
    _write_rollup_for_variant(
        app_root=app_root,
        run_group=run_group_r01,
        queue_rows=rows_r01,
        risk_tag="risk0p02",
    )

    # 3) Plan next again: consumes completed r01 (selection+state advance) and emits r02 queue.
    rc2 = _run_main_isolated(mod, ["--config", "configs/autopilot/test.yaml", "--resume", "--plan-next"])
    assert rc2 == 0

    run_group_r02 = f"{run_group_prefix}_r02_risk"
    queue_r02 = app_root / "artifacts" / "wfa" / "aggregate" / run_group_r02 / "run_queue.csv"
    assert queue_r02.exists()

    state_path = app_root / "artifacts" / "wfa" / "aggregate" / controller_group / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert int(state.get("round") or 0) == 2
    assert "risk0p02" in str(state.get("base_config_path") or "")
