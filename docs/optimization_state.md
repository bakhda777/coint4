# Optimization state

Last updated: 2026-01-16

Current stage: Piogoga grid (leader filters, zscore sweep) via artifacts/wfa/aggregate/20260116_piogoga_grid/run_queue.csv (parallel=16, n_jobs=1)

Progress:
- ssd5000 completed
- ssd10000 completed
- ssd25000 completed
- ssd50000 active (WF step 3/3, step 2: 122 pairs, P&L +355.58)

Parallel stage:
- Signal grid (16 configs, z=0.75/0.8/0.85/0.9 × exit=0.04/0.06/0.08/0.1) via artifacts/wfa/aggregate/20260116_signal_grid/run_queue.csv (parallel=16, n_jobs=1).
- SSD sweep (6 values) via artifacts/wfa/aggregate/20260115_ssd_topn_sweep/run_queue.csv (queue has running statuses; check worker activity before resume).
- Leader validation (post-analysis, single run) queued: artifacts/wfa/aggregate/20260116_leader_validation/run_queue.csv (parallel=1, n_jobs=-1).
- Патч: фильтрация пар теперь параллельная (n_jobs из backtest, processes при memory-map) — ожидаем полную загрузку CPU.

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
