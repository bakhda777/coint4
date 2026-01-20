# Журнал прогонов оптимизации (2026-01-20)

Назначение: нормализовать учет издержек, отфильтровать zero‑cost прогоны, зафиксировать shortlist для 5‑step WFA и stress‑прогонов.

## Обновления (2026-01-20)

### Rollup и gating
- Rollup обновлен: `coint4/artifacts/wfa/aggregate/rollup/run_index.csv` (всего 474 прогонов, метрики: 362).
- Прогоны с издержками `total_costs` = NA: 116; с сделками и zero‑cost: 137.
- Gating (для shortlist): `total_costs>0`, `total_trades>=500`, `total_pairs_traded>=50`, `sharpe_ratio_abs>0`, `total_pnl>0` → кандидатов: 147.
- Артефакт shortlist: `coint4/artifacts/wfa/aggregate/20260120_realcost_shortlist/shortlist_20260120.csv`.

### Shortlist (holdout, costs>0)
| run_id | sharpe | pnl | costs | trades | pairs | max_dd | cost_ratio |
|---|---|---|---|---|---|---|---|
| holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p2 | 9.09 | 1135.3 | 326.4 | 11384 | 120 | -82.2 | 0.29 |
| holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1 | 9.01 | 1114.7 | 326.8 | 11414 | 120 | -82.1 | 0.29 |
| holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p15 | 8.95 | 1107.5 | 326.5 | 11400 | 120 | -80.3 | 0.29 |
| holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z0p95_exit0p08_hold180_cd180_ms0p1 | 8.02 | 1179.8 | 351.3 | 12367 | 120 | -87.7 | 0.30 |
| holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_z0p90_exit0p08_hold120_cd120_ms0p1 | 7.98 | 1071.6 | 650.4 | 25338 | 168 | -93.6 | 0.61 |
| holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z0p90_exit0p06_hold180_cd180_ms0p1 | 7.90 | 1106.6 | 349.3 | 12191 | 120 | -68.7 | 0.32 |

### Изменения конфигов (realistic costs)
Включено `enable_realistic_costs: true` для shortlist:
- coint4/configs/holdout_20260123_relaxed8_nokpss_u250_churnfix_top50_churngrid/holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p2.yaml
- coint4/configs/holdout_20260122_relaxed8_nokpss_u250_churnfix_top50_sens/holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1.yaml
- coint4/configs/holdout_20260123_relaxed8_nokpss_u250_churnfix_top50_churngrid/holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p15.yaml
- coint4/configs/holdout_20260122_relaxed8_nokpss_u250_churnfix_top50_sens/holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z0p95_exit0p08_hold180_cd180_ms0p1.yaml
- coint4/configs/holdout_20260122_relaxed8_nokpss_u250_churnfix/holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_z0p90_exit0p08_hold120_cd120_ms0p1.yaml
- coint4/configs/holdout_20260122_relaxed8_nokpss_u250_churnfix_top50_sens/holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z0p90_exit0p06_hold180_cd180_ms0p1.yaml

### Queue: realcost_shortlist (holdout + stress)
- Очередь: `coint4/artifacts/wfa/aggregate/20260120_realcost_shortlist/run_queue.csv` (12 прогонов).
- Запуск только на 85.198.90.128 (здесь не запускать):
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py --queue artifacts/wfa/aggregate/20260120_realcost_shortlist/run_queue.csv --parallel $(nproc)`

### Следующие шаги
- Запустить 5‑step WFA на 85.198.90.128 для shortlist (holdout + stress) с теми же издержками.
- Проверить устойчивость Sharpe/PnL по шагам и стресс‑сценариям; исключить конфиги с cost_ratio > 0.5 и нестабильной доходностью.
