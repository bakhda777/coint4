from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_rank_module(tmp_path: Path):
    script_path = Path(__file__).resolve().parents[2] / "scripts/optimization/rank_multiwindow_robust_runs.py"
    spec = importlib.util.spec_from_file_location(f"rank_multiwindow_test_{tmp_path.name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_daily_pnl(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "PnL"])
        for idx, pnl in enumerate(values, start=1):
            writer.writerow([f"2024-01-{idx:02d}", pnl])


def _run_index_fieldnames() -> list[str]:
    return [
        "run_id",
        "run_group",
        "results_dir",
        "config_path",
        "status",
        "metrics_present",
        "sharpe_ratio_abs",
        "psr",
        "dsr",
        "total_pnl",
        "max_drawdown_on_equity",
        "total_trades",
        "total_pairs_traded",
        "max_drawdown_abs",
        "total_costs",
        "tail_loss_worst_pair_share",
        "tail_loss_worst_period_share",
    ]


def _build_run_index(tmp_path: Path) -> Path:
    run_group = "rg_fullspan"
    rows: list[dict[str, str]] = []

    def add_pair(
        *,
        base_id: str,
        holdout_sharpe: float,
        stress_sharpe: float,
        holdout_psr: float,
        stress_psr: float,
        holdout_dsr: float,
        stress_dsr: float,
        holdout_pnl: float,
        stress_pnl: float,
        holdout_dd_pct: float,
        stress_dd_pct: float,
        holdout_daily: list[float],
        stress_daily: list[float],
    ) -> None:
        holdout_results = tmp_path / "runs" / f"holdout_{base_id}"
        stress_results = tmp_path / "runs" / f"stress_{base_id}"
        _write_daily_pnl(holdout_results / "daily_pnl.csv", holdout_daily)
        _write_daily_pnl(stress_results / "daily_pnl.csv", stress_daily)

        rows.append(
            {
                "run_id": f"holdout_{base_id}",
                "run_group": run_group,
                "results_dir": str(holdout_results),
                "config_path": f"configs/{base_id}_holdout.yaml",
                "status": "completed",
                "metrics_present": "true",
                "sharpe_ratio_abs": str(holdout_sharpe),
                "psr": str(holdout_psr),
                "dsr": str(holdout_dsr),
                "total_pnl": str(holdout_pnl),
                "max_drawdown_on_equity": str(holdout_dd_pct),
                "total_trades": "500",
                "total_pairs_traded": "30",
                "max_drawdown_abs": "-100",
                "total_costs": "10",
            }
        )
        rows.append(
            {
                "run_id": f"stress_{base_id}",
                "run_group": run_group,
                "results_dir": str(stress_results),
                "config_path": f"configs/{base_id}_stress.yaml",
                "status": "completed",
                "metrics_present": "true",
                "sharpe_ratio_abs": str(stress_sharpe),
                "psr": str(stress_psr),
                "dsr": str(stress_dsr),
                "total_pnl": str(stress_pnl),
                "max_drawdown_on_equity": str(stress_dd_pct),
                "total_trades": "500",
                "total_pairs_traded": "30",
                "max_drawdown_abs": "-100",
                "total_costs": "10",
            }
        )

    # Better robust Sharpe, but catastrophic left tail on robust daily PnL.
    add_pair(
        base_id="variant_a",
        holdout_sharpe=1.50,
        stress_sharpe=1.20,
        holdout_psr=0.42,
        stress_psr=0.35,
        holdout_dsr=-0.30,
        stress_dsr=-0.50,
        holdout_pnl=120.0,
        stress_pnl=100.0,
        holdout_dd_pct=0.10,
        stress_dd_pct=0.12,
        holdout_daily=[12.0, -320.0, 20.0],
        stress_daily=[8.0, -300.0, 9.0],
    )
    # Lower robust Sharpe, but acceptable tail profile.
    add_pair(
        base_id="variant_b",
        holdout_sharpe=1.00,
        stress_sharpe=0.95,
        holdout_psr=0.96,
        stress_psr=0.93,
        holdout_dsr=0.40,
        stress_dsr=0.25,
        holdout_pnl=80.0,
        stress_pnl=70.0,
        holdout_dd_pct=0.08,
        stress_dd_pct=0.10,
        holdout_daily=[-55.0, 20.0, 15.0],
        stress_daily=[-50.0, 11.0, 12.0],
    )

    run_index = tmp_path / "run_index.csv"
    fieldnames = _run_index_fieldnames()
    with run_index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return run_index


def _build_run_index_for_score_modes(tmp_path: Path) -> Path:
    run_group = "rg_modes"
    rows: list[dict[str, str]] = []
    windows = [
        ("20240101", "20240301"),
        ("20240301", "20240501"),
        ("20240501", "20240701"),
        ("20240701", "20240901"),
        ("20240901", "20241101"),
    ]
    variant_a_robust = [0.20, 1.40, 1.40, 1.40, 1.40]
    variant_b_robust = [0.60, 1.00, 1.00, 1.00, 1.00]

    def add_variant_window(variant: str, start: str, end: str, robust_sharpe: float) -> None:
        base_id = f"{variant}_oos{start}_{end}"
        holdout_results = tmp_path / "runs" / f"holdout_{base_id}"
        stress_results = tmp_path / "runs" / f"stress_{base_id}"
        holdout_results.mkdir(parents=True, exist_ok=True)
        stress_results.mkdir(parents=True, exist_ok=True)

        rows.append(
            {
                "run_id": f"holdout_{base_id}",
                "run_group": run_group,
                "results_dir": str(holdout_results),
                "config_path": f"configs/{base_id}_holdout.yaml",
                "status": "completed",
                "metrics_present": "true",
                "sharpe_ratio_abs": str(robust_sharpe + 0.10),
                "psr": "0.95",
                "dsr": "0.20",
                "total_pnl": "100.0",
                "max_drawdown_on_equity": "0.10",
                "total_trades": "500",
                "total_pairs_traded": "30",
                "max_drawdown_abs": "-100",
                "total_costs": "10",
            }
        )
        rows.append(
            {
                "run_id": f"stress_{base_id}",
                "run_group": run_group,
                "results_dir": str(stress_results),
                "config_path": f"configs/{base_id}_stress.yaml",
                "status": "completed",
                "metrics_present": "true",
                "sharpe_ratio_abs": str(robust_sharpe),
                "psr": "0.95",
                "dsr": "0.20",
                "total_pnl": "90.0",
                "max_drawdown_on_equity": "0.11",
                "total_trades": "500",
                "total_pairs_traded": "30",
                "max_drawdown_abs": "-100",
                "total_costs": "10",
            }
        )

    for (start, end), robust in zip(windows, variant_a_robust):
        add_variant_window("variant_a", start, end, robust)
    for (start, end), robust in zip(windows, variant_b_robust):
        add_variant_window("variant_b", start, end, robust)

    run_index = tmp_path / "run_index_modes.csv"
    fieldnames = _run_index_fieldnames()
    with run_index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return run_index


def _build_run_index_for_concentration_gates(tmp_path: Path) -> Path:
    run_group = "rg_concentration"
    rows: list[dict[str, str]] = []

    def _share(value: float | None) -> str:
        return "" if value is None else str(value)

    def add_pair(
        *,
        base_id: str,
        holdout_sharpe: float,
        stress_sharpe: float,
        holdout_pair_share: float | None,
        stress_pair_share: float | None,
        holdout_period_share: float | None,
        stress_period_share: float | None,
    ) -> None:
        holdout_results = tmp_path / "runs" / f"holdout_{base_id}"
        stress_results = tmp_path / "runs" / f"stress_{base_id}"
        holdout_results.mkdir(parents=True, exist_ok=True)
        stress_results.mkdir(parents=True, exist_ok=True)

        rows.append(
            {
                "run_id": f"holdout_{base_id}",
                "run_group": run_group,
                "results_dir": str(holdout_results),
                "config_path": f"configs/{base_id}_holdout.yaml",
                "status": "completed",
                "metrics_present": "true",
                "sharpe_ratio_abs": str(holdout_sharpe),
                "psr": "0.95",
                "dsr": "0.20",
                "total_pnl": "120.0",
                "max_drawdown_on_equity": "0.11",
                "total_trades": "500",
                "total_pairs_traded": "30",
                "max_drawdown_abs": "-100",
                "total_costs": "10",
                "tail_loss_worst_pair_share": _share(holdout_pair_share),
                "tail_loss_worst_period_share": _share(holdout_period_share),
            }
        )
        rows.append(
            {
                "run_id": f"stress_{base_id}",
                "run_group": run_group,
                "results_dir": str(stress_results),
                "config_path": f"configs/{base_id}_stress.yaml",
                "status": "completed",
                "metrics_present": "true",
                "sharpe_ratio_abs": str(stress_sharpe),
                "psr": "0.95",
                "dsr": "0.20",
                "total_pnl": "100.0",
                "max_drawdown_on_equity": "0.12",
                "total_trades": "500",
                "total_pairs_traded": "30",
                "max_drawdown_abs": "-100",
                "total_costs": "10",
                "tail_loss_worst_pair_share": _share(stress_pair_share),
                "tail_loss_worst_period_share": _share(stress_period_share),
            }
        )

    add_pair(
        base_id="variant_a",
        holdout_sharpe=1.30,
        stress_sharpe=1.20,
        holdout_pair_share=0.92,
        stress_pair_share=0.88,
        holdout_period_share=0.58,
        stress_period_share=0.56,
    )
    add_pair(
        base_id="variant_b",
        holdout_sharpe=1.00,
        stress_sharpe=0.95,
        holdout_pair_share=0.35,
        stress_pair_share=0.30,
        holdout_period_share=0.40,
        stress_period_share=0.42,
    )
    add_pair(
        base_id="variant_c",
        holdout_sharpe=1.10,
        stress_sharpe=1.05,
        holdout_pair_share=0.30,
        stress_pair_share=0.28,
        holdout_period_share=None,
        stress_period_share=None,
    )

    run_index = tmp_path / "run_index_concentration.csv"
    with run_index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_run_index_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return run_index


def _first_rank_row(output: str) -> dict[str, str]:
    lines = [line for line in output.splitlines() if line.startswith("|")]
    if len(lines) < 3:
        raise AssertionError(f"markdown table not found in output:\n{output}")

    header = [chunk.strip() for chunk in lines[0].strip("|").split("|")]
    for line in lines[2:]:
        cols = [chunk.strip() for chunk in line.strip("|").split("|")]
        if len(cols) != len(header):
            continue
        row = dict(zip(header, cols))
        if row.get("rank") == "1":
            return row
    raise AssertionError("rank row #1 not found")


def test_default_mode_prefers_higher_worst_robust_sharpe(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load_rank_module(tmp_path)
    run_index = _build_run_index(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rank_multiwindow_robust_runs.py",
            "--run-index",
            str(run_index),
            "--min-windows",
            "1",
            "--min-trades",
            "1",
            "--min-pairs",
            "1",
            "--top",
            "2",
        ],
    )
    rc = module.main()
    assert rc == 0

    top_row = _first_rank_row(capsys.readouterr().out)
    assert top_row["variant_id"] == "variant_a"
    assert top_row["sample_config"].endswith("variant_a_holdout.yaml")
    assert top_row["score_mode"] == "worst"


def test_psr_dsr_gate_rejects_false_leader(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load_rank_module(tmp_path)
    run_index = _build_run_index(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rank_multiwindow_robust_runs.py",
            "--run-index",
            str(run_index),
            "--min-windows",
            "1",
            "--min-trades",
            "1",
            "--min-pairs",
            "1",
            "--min-psr",
            "0.9",
            "--min-dsr",
            "0.0",
            "--top",
            "2",
        ],
    )
    rc = module.main()
    assert rc == 0

    top_row = _first_rank_row(capsys.readouterr().out)
    assert top_row["variant_id"] == "variant_b"
    assert float(top_row["worst_psr"]) >= 0.9
    assert float(top_row["worst_dsr"]) >= 0.0


def test_fullspan_policy_v1_rejects_bad_tail_and_promotes_safer_variant(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _load_rank_module(tmp_path)
    run_index = _build_run_index(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rank_multiwindow_robust_runs.py",
            "--run-index",
            str(run_index),
            "--min-windows",
            "1",
            "--min-trades",
            "1",
            "--min-pairs",
            "1",
            "--top",
            "2",
            "--fullspan-policy-v1",
            "--initial-capital",
            "1000",
            "--tail-worst-gate-pct",
            "0.20",
        ],
    )
    rc = module.main()
    assert rc == 0

    output = capsys.readouterr().out
    top_row = _first_rank_row(output)
    assert top_row["variant_id"] == "variant_b"
    assert "variant_a" not in output


def test_quantile_mode_prefers_quantile_stable_variant(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load_rank_module(tmp_path)
    run_index = _build_run_index_for_score_modes(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rank_multiwindow_robust_runs.py",
            "--run-index",
            str(run_index),
            "--min-windows",
            "5",
            "--min-trades",
            "1",
            "--min-pairs",
            "1",
            "--top",
            "2",
            "--score-mode",
            "quantile",
            "--quantile-q",
            "0.20",
        ],
    )
    rc = module.main()
    assert rc == 0

    top_row = _first_rank_row(capsys.readouterr().out)
    assert top_row["score_mode"] == "quantile"
    assert top_row["variant_id"] == "variant_a"
    assert float(top_row["q20_robust_sh"]) > float(top_row["worst_robust_sh"])


def test_hybrid_mode_uses_weighted_score_formula(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load_rank_module(tmp_path)
    run_index = _build_run_index_for_score_modes(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rank_multiwindow_robust_runs.py",
            "--run-index",
            str(run_index),
            "--min-windows",
            "5",
            "--min-trades",
            "1",
            "--min-pairs",
            "1",
            "--top",
            "2",
            "--score-mode",
            "hybrid",
            "--quantile-q",
            "0.20",
            "--hybrid-worst-weight",
            "0.50",
            "--hybrid-quantile-weight",
            "0.30",
            "--hybrid-avg-weight",
            "0.20",
        ],
    )
    rc = module.main()
    assert rc == 0

    top_row = _first_rank_row(capsys.readouterr().out)
    assert top_row["score_mode"] == "hybrid"
    assert top_row["variant_id"] == "variant_b"
    expected = (
        0.50 * float(top_row["worst_robust_sh"])
        + 0.30 * float(top_row["q20_robust_sh"])
        + 0.20 * float(top_row["avg_robust_sh"])
    )
    assert float(top_row["score"]) == pytest.approx(expected, abs=1e-3)


def test_pair_concentration_gate_rejects_overconcentrated_variant(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _load_rank_module(tmp_path)
    run_index = _build_run_index_for_concentration_gates(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rank_multiwindow_robust_runs.py",
            "--run-index",
            str(run_index),
            "--min-windows",
            "1",
            "--min-trades",
            "1",
            "--min-pairs",
            "1",
            "--max-tail-pair-share",
            "0.60",
            "--top",
            "3",
        ],
    )
    rc = module.main()
    assert rc == 0

    output = capsys.readouterr().out
    top_row = _first_rank_row(output)
    assert top_row["variant_id"] == "variant_c"
    assert "Concentration gate rejections:" in output
    assert "pair_above_max=1" in output


def test_period_concentration_gate_fail_closed_on_missing_values(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _load_rank_module(tmp_path)
    run_index = _build_run_index_for_concentration_gates(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rank_multiwindow_robust_runs.py",
            "--run-index",
            str(run_index),
            "--min-windows",
            "1",
            "--min-trades",
            "1",
            "--min-pairs",
            "1",
            "--max-tail-period-share",
            "0.50",
            "--top",
            "3",
        ],
    )
    rc = module.main()
    assert rc == 0

    output = capsys.readouterr().out
    top_row = _first_rank_row(output)
    assert top_row["variant_id"] == "variant_b"
    assert "Concentration gate rejections:" in output
    assert "period_above_max=1" in output
    assert "period_missing=1" in output
