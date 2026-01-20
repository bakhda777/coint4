# Журнал прогонов оптимизации (2026-01-22)

Назначение: churn-control micro-grid после включения cooldown/min_hold/min_spread_move, проверка компромисса Sharpe/PnL/turnover.

## Обновления (2026-01-22)

### Queue: relaxed8_nokpss_u250_churnfix (holdout + stress)
- Очередь: `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix/run_queue.csv`.
- Цель: проверить churn-контроль (min_hold, cooldown, min_spread_move_sigma, entry/exit, max_active_positions).
- Конфиги:
  - `coint4/configs/holdout_20260122_relaxed8_nokpss_u250_churnfix/*.yaml` (8 шт.)
  - `coint4/configs/stress_20260122_relaxed8_nokpss_u250_churnfix/*.yaml` (8 шт.)
- Статус: `completed` (16 прогонов).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| z0p90/exit0p08/hold120/cd120/ms0p1 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p06/hold120/cd120/ms0p1 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p06/hold120/cd120/ms0p1/maxpos10 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p06/hold120/cd120/ms0p2 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p06/hold180/cd180/ms0p1 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p06/hold60/cd60/ms0p1 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p10/hold120/cd120/ms0p1 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z1p00/exit0p06/hold120/cd120/ms0p1 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |

Выводы:
- Все варианты дали 0 сделок; причина — слишком консервативные адаптивные пороги в Numba (мультипликатор упирался в max).
- Исправление: адаптивные пороги переведены на base-volatility + `max_var_multiplier` (см. `coint4/src/coint2/core/numba_kernels.py`).
- Повторный прогон queued: `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_v2/run_queue.csv`.
