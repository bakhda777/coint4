# Optimization runs — 2026-02-20 (baseline refresh before next loop)

Контекст: перед новым loop нужен свежий baseline на актуальном rollup и отдельный снимок clean-кандидатов.

## Rebuild rollup (including `runs_clean`)

Команда (из `coint4/`):

`PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`

Результат:
- `Run index entries: 1800`.
- Покрытие `runs_clean` в `run_index.csv`: `20` строк, `20/20 status=completed`, `20/20 metrics_present=true`.

## Baseline snapshot: top robust candidates (global multi-window)

Команда (из `coint4/`):

`PYTHONPATH=src ./.venv/bin/python scripts/optimization/rank_multiwindow_robust_runs.py --top 10 --max-dd-pct 0.14 --min-windows 3 --min-trades 200 --min-pairs 20 --contains budget1000 --include-noncompleted`

Результат (top-10):

| rank | worst_robust_sh | avg_robust_sh | worst_dd_pct | avg_dd_pct | windows | run_group | variant_id |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 3.530 | 4.388 | 0.132 | 0.089 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p91 |
| 2 | 3.530 | 4.388 | 0.132 | 0.089 | 3 | 20260214_budget1000_dd_sprint09_hurst_slusd1p91 | prod_final_budget1000_risk0p006_slusd1p91_max_hurst_exponent0p8 |
| 3 | 3.470 | 4.357 | 0.128 | 0.088 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p88 |
| 4 | 3.448 | 4.306 | 0.132 | 0.089 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p85_risk0p006 |
| 5 | 3.448 | 4.306 | 0.132 | 0.089 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p85 |
| 6 | 3.323 | 4.254 | 0.137 | 0.087 | 3 | 20260213_budget1000_dd_sprint05_stoplossusd_refine | pruned168_slusd1p75_risk0p006 |
| 7 | 3.323 | 4.254 | 0.137 | 0.087 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p75_risk0p006 |
| 8 | 3.299 | 4.165 | 0.138 | 0.091 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p8_risk0p006 |
| 9 | 3.179 | 4.200 | 0.137 | 0.086 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p65_risk0p006 |
| 10 | 2.864 | 4.115 | 0.108 | 0.073 | 3 | 20260213_budget1000_dd_sprint05_stoplossusd_refine | pruned168_slusd1p5_risk0p006 |

Baseline freeze для следующего loop:
- Baseline score (worst robust Sharpe): `3.530254`.
- Baseline worst_dd_pct: `0.132205`.
- Baseline run_group: `20260213_budget1000_dd_sprint08_stoplossusd_micro`.
- Baseline run_id (worst-window holdout): `holdout_prod_final_budget1000_oos20240501_20250630_risk0p006_slusd1p91`.
- Baseline paired stress run_id (worst-window): `stress_prod_final_budget1000_oos20240501_20250630_risk0p006_slusd1p91`.
- Baseline variant_id: `prod_final_budget1000_risk0p006_slusd1p91`.
- Gate check: `worst_dd_pct=0.132205 <= 0.14`, `windows=3`, `min_trades>=200`, `min_pairs>=20`.
- Tie-break decision: у `20260214_budget1000_dd_sprint09_hurst_slusd1p91` тот же `worst_robust_sharpe`; для continuity baseline оставлен на `20260213...sprint08`.

## Snapshot: top clean candidates (`runs_clean`)

Команда (из `coint4/`):

`PYTHONPATH=src ./.venv/bin/python scripts/optimization/rank_robust_runs.py --top 10 --contains runs_clean --contains confirm_shortlist`

Результат (top-5):

| rank | robust_sharpe | run_group | run_id | holdout_pnl | stress_pnl | note |
|---:|---:|---|---|---:|---:|---|
| 1 | 0.000 | 20260215_confirm_shortlist | b01_20260120_realcost_shortlist_..._ms0p2 | 0.00 | 0.00 | no-op |
| 2 | 0.000 | 20260215_confirm_shortlist | b02_20260123_relaxed8_nokpss_u250_churnfix_..._ms0p2 | 0.00 | 0.00 | no-op |
| 3 | 0.000 | 20260215_confirm_shortlist | b03_20260120_realcost_shortlist_..._ms0p1 | 0.00 | 0.00 | no-op |
| 4 | 0.000 | 20260215_confirm_shortlist | b04_20260122_relaxed8_nokpss_u250_churnfix_..._ms0p1 | 0.00 | 0.00 | no-op |
| 5 | 0.000 | 20260215_confirm_shortlist | b05_20260120_realcost_shortlist_..._ms0p15 | 0.00 | 0.00 | no-op |

Вывод по clean snapshot:
- `10/10` paired holdout/stress в `runs_clean` имеют `robust_sharpe=0`, `PnL=0`, `DD=0`.
- До выяснения причины no-op clean-shortlist не используется как baseline для запуска нового loop.
