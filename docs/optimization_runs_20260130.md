# Журнал прогонов оптимизации (2026-01-30)

Назначение: extended OOS (2023-05-01 -> 2024-04-30) для top20/top30 кандидатов с realistic costs.

## Критерии отбора (актуальные)
- Гейтинг: `total_costs>0`, `total_trades>=500`, `total_pairs_traded>=50`, `sharpe_ratio_abs>0`, `total_pnl>0`.
- Stress: `cost_ratio <= 0.5` (издержки не более 50% PnL).
- WFA-стабильность: медиана Sharpe по 5 шагам >= 1.0, минимум по шагам >= 0.6.

## Очередь: realcost_oos20230501_20240430
- Очередь: `coint4/artifacts/wfa/aggregate/20260130_realcost_oos20230501_20240430/run_queue.csv` (4 прогона).
- Конфиги:
  - `coint4/configs/holdout_20260130_relaxed8_nokpss_u250_churnfix_oos20230501_20240430/holdout_relaxed8_nokpss_20260130_oos20230501_20240430_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2.yaml`
  - `coint4/configs/stress_20260130_relaxed8_nokpss_u250_churnfix_oos20230501_20240430/stress_relaxed8_nokpss_20260130_oos20230501_20240430_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2.yaml`
  - `coint4/configs/holdout_20260130_relaxed8_nokpss_u250_churnfix_oos20230501_20240430/holdout_relaxed8_nokpss_20260130_oos20230501_20240430_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top20_z1p00_exit0p06_hold180_cd180_ms0p2.yaml`
  - `coint4/configs/stress_20260130_relaxed8_nokpss_u250_churnfix_oos20230501_20240430/stress_relaxed8_nokpss_20260130_oos20230501_20240430_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top20_z1p00_exit0p06_hold180_cd180_ms0p2.yaml`
- Статус: `planned`.

## План
- Запустить WFA (<=5 шагов) только на 85.198.90.128 с `--parallel $(nproc)`.
- Обновить rollup и зафиксировать результаты/выводы.

## Результаты
- TBA.
