# Optimization state

Last updated: 2026-01-19

Current stage: Sharpe>1 program (см. docs/optimization_plan_20260118.md). Базовые WFA/sweep завершены, shortlist WFA выполнен; далее holdout + стресс-издержки для топ-1/2.

Next steps:
- Исправление `total_costs` для Numba-бэктеста выполнено; метрики обновлены (done).
- Baseline WFA (5 шагов) выполнен и зафиксирован (done).
- Turnover sweep завершён, лучшая комбинация entry 0.95 / exit 0.10 (done).
- Quality sweep завершён, лучший Sharpe при corr 0.65 (done).
- Risk sweep завершён, существенных отличий не выявлено (done).
- Shortlist WFA (5 шагов) завершён на 85.198.90.128; топ Sharpe: baseline `5.7560`, corr0.7 `5.7302`, turnover `4.9631` (см. rollup).
- Holdout WFA (2024-05-01 → 2024-12-31, фактический тест до 2024-09-28) завершён: Sharpe `-3.41/-3.27`, PnL `-324/-307` (baseline/corr0.7).
- Stress costs WFA завершён: Sharpe `4.66/4.57`, PnL `616/543` (baseline/corr0.7).
- Диагностика holdout завершена: см. `docs/holdout_diagnostics_20260118.md` + CSV в `coint4/results/holdout_20260118_*`.
- Добавлен фильтр стабильности пар (pair_stability_window_steps/min_steps) в WFA.
- Stability shortlist (20260119) завершён: Sharpe 2.41–5.92, но total_pairs_traded 4–34 (ниже порога 100).
- Stability_relaxed WFA завершён: Sharpe `3.99/2.59`, total_pairs_traded `78/41` (ещё ниже порога 100).
- Stability_relaxed2 WFA завершён: Sharpe `2.13`, total_pairs_traded `52`, total_trades `1010`, PnL `84.13`.
- Stability_relaxed3 WFA завершён: лучший Sharpe `2.04`, total_pairs_traded `51`; 2 конфига дали 0 пар из-за KPSS фильтра.
- Далее: ослабить KPSS (например, 0.03–0.05) и повторить corr0.5 + ssd 40000/50000; при необходимости снизить corr до 0.45.

Legacy context:
Current stage: Leader holdout WFA (2024-05-01 → 2024-12-31, max_steps=5) via artifacts/wfa/aggregate/20260116_leader_holdout/run_queue.csv (parallel=1, n_jobs=-1). Additional: next5_fast WFA (manual sequential runs; queue file artifacts/wfa/aggregate/20260117_next5_fast/run_queue_next5_fast.csv used for status, backtest.n_jobs=-1, COINT_FILTER_BACKEND=threads). Current next5_fast run: none (latest best by Sharpe: pair_sweep_20260117_corr0p55_z0p85_exit0p12_ssd25000); queued: none.

Progress:
- ssd5000 completed
- ssd10000 completed
- ssd25000 completed
- ssd50000 active (WF step 3/3, step 2: 122 pairs, P&L +355.58)
- leader_validation completed (Sharpe 0.5255, PnL 1388.71, DD -199.31)
- leader_holdout active: coint4/configs/best_config__leader_holdout_ssd25000__20260116_211943.yaml → artifacts/wfa/runs/20260116_leader_holdout/best_config__leader_holdout_ssd25000__20260116_211943 (COINT_FILTER_BACKEND=processes)

Parallel stage:
- Piogoga grid (leader filters, zscore sweep) via artifacts/wfa/aggregate/20260116_piogoga_grid/run_queue.csv (parallel=16, n_jobs=1).
- Signal grid (16 configs, z=0.75/0.8/0.85/0.9 × exit=0.04/0.06/0.08/0.1) via artifacts/wfa/aggregate/20260116_signal_grid/run_queue.csv (parallel=16, n_jobs=1).
- SSD sweep (6 values) via artifacts/wfa/aggregate/20260115_ssd_topn_sweep/run_queue.csv (queue has running statuses; check worker activity before resume).
- Leader validation (post-analysis, single run) completed: artifacts/wfa/runs/20260116_leader_validation/.
- Патч: фильтрация пар теперь параллельная (n_jobs из backtest, backend threads; `COINT_FILTER_BACKEND=processes` с spawn для OpenMP‑безопасности) — цель полной загрузки CPU.

After ssd50000 DONE:
1) Update docs/optimization_runs_20260116.md (metrics + filtering summary for ssd50000).
2) Update rollup in artifacts/wfa/aggregate/rollup/.
3) Start SSD sweep (3 values) queue: artifacts/wfa/aggregate/20260115_ssd_topn_sweep_3vals/run_queue.csv.
4) Update this file with the new stage and next steps.

After signal grid DONE:
1) Update docs/optimization_runs_20260116.md (metrics + filtering summary for signal grid).
2) Update rollup in artifacts/wfa/aggregate/rollup/.

After piogoga grid DONE:
1) Update docs/optimization_runs_20260116.md (metrics + filtering summary for piogoga grid).
2) Update rollup in artifacts/wfa/aggregate/rollup/.
3) Start leader validation queue: artifacts/wfa/aggregate/20260116_leader_validation/run_queue.csv.

After leader holdout DONE:
1) Update docs/optimization_runs_20260116.md (metrics + filtering summary for leader holdout).
2) Update rollup in artifacts/wfa/aggregate/rollup/.
3) Decide next stage (risk sweep vs signal grid refinement) based on holdout Sharpe/PnL/DD.

Notes:
- NOTE: Sharpe в записях/артефактах до фикса annualization (2026-01-18) занижен примерно в √96 раз для 15m; для актуальных значений используйте `coint4/artifacts/wfa/aggregate/rollup/run_index.*`.
- 2026-01-18: shortlist WFA completed на 85.198.90.128, артефакты синхронизированы, сервер выключен.
- 2026-01-18: holdout + stress WFA завершены на 85.198.90.128, артефакты синхронизированы, сервер выключен.
- 2026-01-17: smoke WFA для проверки логирования команд (config main_2024_smoke.yaml, results artifacts/wfa/runs/logging_smoke_20260117_072821).
- 2026-01-17: next5_fast completed for signal_sweep_20260116_z0p85_exit0p06_ssd25000 (PnL 815.67, Sharpe 0.6345, DD -132.02).
- 2026-01-17: next5_fast completed for signal_sweep_20260116_z0p85_exit0p08_ssd25000 (PnL 821.12, Sharpe 0.6410, DD -128.32).
- 2026-01-17: next5_fast completed for risk_sweep_20260116_stop2p5_time3p5 (PnL 771.63, Sharpe 0.5860, DD -146.72).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p9_exit0p08_ssd25000 (PnL 741.27, Sharpe 0.5908, DD -146.05).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p9_exit0p1_ssd25000 (PnL 742.94, Sharpe 0.5933, DD -146.08).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p95_exit0p08_ssd25000 (PnL 685.02, Sharpe 0.5665, DD -114.16).
- 2026-01-17: next5_fast completed for risk_sweep_20260117_stop2p5_time2p0_z0p85_exit0p08_ssd25000 (PnL 821.12, Sharpe 0.6410, DD -128.32).
- 2026-01-17: next5_fast completed for risk_sweep_20260117_stop3p0_time2p5_z0p85_exit0p08_ssd25000 (PnL 821.12, Sharpe 0.6410, DD -128.32).
- 2026-01-17: next5_fast completed for risk_sweep_20260117_stop3p5_time3p0_z0p85_exit0p08_ssd25000 (PnL 821.12, Sharpe 0.6410, DD -128.32).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p85_exit0p06_ssd25000 (PnL 815.67, Sharpe 0.6345, DD -132.02).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p85_exit0p1_ssd25000 (PnL 821.86, Sharpe 0.6434, DD -124.82).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p8_exit0p06_ssd25000 (PnL 771.63, Sharpe 0.5860, DD -146.72).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p85_exit0p12_ssd25000 (PnL 855.78, Sharpe 0.6789, DD -95.32).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p85_exit0p09_ssd25000 (PnL 822.87, Sharpe 0.6441, DD -124.82).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p9_exit0p1_ssd25000 (PnL 742.94, Sharpe 0.5933, DD -146.08).
- 2026-01-17: next5_fast completed for signal_sweep_20260117_z0p8_exit0p1_ssd25000 (PnL 780.90, Sharpe 0.5965, DD -139.51).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p6_z0p85_exit0p12_ssd25000 (PnL 666.32, Sharpe 0.7452, DD -95.32).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_pv0p03_top800_z0p85_exit0p12_ssd25000 (PnL 789.56, Sharpe 0.6637, DD -98.47).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_hurst0p52_z0p85_exit0p12_ssd25000 (PnL 744.46, Sharpe 0.6037, DD -90.68).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_ssd15000_z0p85_exit0p12_ssd25000 (PnL 404.72, Sharpe 0.6275, DD -44.95).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p6_pv0p03_top800_z0p85_exit0p12_ssd25000 (PnL 621.19, Sharpe 0.7190, DD -91.63).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p6_hurst0p52_z0p85_exit0p12_ssd25000 (PnL 555.34, Sharpe 0.6492, DD -90.68).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_pv0p03_top800_kpss0p03_z0p85_exit0p12_ssd25000 (PnL 946.25, Sharpe 0.5384, DD -193.91).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p55_z0p85_exit0p12_ssd25000 (PnL 666.32, Sharpe 0.7452, DD -95.32).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p7_pv0p02_top500_z0p85_exit0p12_ssd25000 (PnL 456.81, Sharpe 0.6618, DD -78.36).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p65_hurst0p5_kpss0p03_z0p85_exit0p12_ssd25000 (PnL 620.39, Sharpe 0.5028, DD -126.15).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_ssd8000_pv0p03_top600_z0p85_exit0p12_ssd25000 (PnL 247.31, Sharpe 0.4533, DD -37.73).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_hl0p1_30_corr0p6_z0p85_exit0p12_ssd25000 (PnL 547.88, Sharpe 0.6830, DD -95.32).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p75_pv0p015_top300_hurst0p48_kpss0p03_z0p85_exit0p12 (PnL 95.37, Sharpe 0.1718, DD -105.85).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p7_pv0p01_top200_kpss0p02_hl0p2_20_z0p85_exit0p12 (PnL 623.38, Sharpe 0.4579, DD -230.50).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_ssd4000_corr0p65_pv0p02_top400_z0p85_exit0p12 (PnL 61.89, Sharpe 0.2783, DD -25.44).
- 2026-01-17: next5_fast completed for pair_sweep_20260117_corr0p7_hurst0p5_kpss0p02_cross1_z0p85_exit0p12 (PnL 747.15, Sharpe 0.5444, DD -130.58).
