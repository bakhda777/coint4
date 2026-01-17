# Optimization state

Last updated: 2026-01-17

Current stage: Leader holdout WFA (2024-05-01 → 2024-12-31, max_steps=5) via artifacts/wfa/aggregate/20260116_leader_holdout/run_queue.csv (parallel=1, n_jobs=-1). Additional: next5_fast WFA (manual sequential runs; queue file artifacts/wfa/aggregate/20260117_next5_fast/run_queue_next5_fast.csv used for status, backtest.n_jobs=-1, COINT_FILTER_BACKEND=threads). Current next5_fast run: none (latest best: signal_sweep_20260117_z0p85_exit0p12_ssd25000); queued: pair_sweep_20260117_corr0p6_z0p85_exit0p12_ssd25000, pair_sweep_20260117_pv0p03_top800_z0p85_exit0p12_ssd25000, pair_sweep_20260117_hurst0p52_z0p85_exit0p12_ssd25000, pair_sweep_20260117_ssd15000_z0p85_exit0p12_ssd25000.

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
