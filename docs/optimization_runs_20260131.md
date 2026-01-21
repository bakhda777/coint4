# Журнал прогонов оптимизации (2026-01-31)

Назначение: extended OOS (2023-05-01 -> 2024-04-30) turnover-grid для снижения издержек (max_pairs 10/15, ms 0.25/0.30, hold/cd 240).

## Критерии отбора (актуальные)
- Гейтинг: `total_costs>0`, `total_trades>=500`, `total_pairs_traded>=50`, `sharpe_ratio_abs>0`, `total_pnl>0`.
- Stress: `cost_ratio <= 0.5`.
- WFA-стабильность: медиана Sharpe по 5 шагам >= 1.0, минимум по шагам >= 0.6.

## Очередь: realcost_oos20230501_20240430_turnover
- Очередь: `coint4/artifacts/wfa/aggregate/20260131_realcost_oos20230501_20240430_turnover/run_queue.csv` (8 прогонов).
- Конфиги:
  - `coint4/configs/holdout_20260131_relaxed8_nokpss_u250_churnfix_turnover_oos20230501_20240430/*.yaml` (4 шт.)
  - `coint4/configs/stress_20260131_relaxed8_nokpss_u250_churnfix_turnover_oos20230501_20240430/*.yaml` (4 шт.)
- Статус: `planned`.

## План
- Запустить WFA (<=5 шагов) только на 85.198.90.128 с `--parallel $(nproc)`.
- Обновить rollup и зафиксировать результаты/выводы.

## Результаты
- TBA.
