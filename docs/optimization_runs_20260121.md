# Журнал прогонов оптимизации (2026-01-21)

Назначение: micro-grid вокруг базового u250 кандидата + max_pairs sweep, проверка Sharpe annualization.

## Обновления (2026-01-21)

### Sharpe annualization sanity
- WFA метрики считают Sharpe через `periods_per_year = annualizing_factor * (24*60/bar_minutes)`.
- Base engine приведён к динамическому periods_per_year по медианному шагу индекса (см. `coint4/src/coint2/engine/base_engine.py`).

### Queue: relaxed8_nokpss_u250_search (micro-grid + max_pairs)
- Очередь: `coint4/artifacts/wfa/aggregate/20260121_relaxed8_nokpss_u250_search/run_queue.csv`.
- Цель: найти локальные улучшения вокруг базового z0.95/0.08/120/120 и проверить компромисс Sharpe/PnL/turnover через `max_pairs`.
- Параллельность: `1` (каждый прогон использует `backtest.n_jobs: -1`).
- Конфиги:
  - `coint4/configs/holdout_20260121_relaxed8_nokpss_u250_search/*.yaml` (9 шт.)
  - `coint4/configs/stress_20260121_relaxed8_nokpss_u250_search/*.yaml` (9 шт.)
- Грид параметров:
  - entry: `0.90`, `0.95`, `1.00`
  - exit: `0.06`, `0.08`, `0.10`
  - hold/cooldown: `90`, `120`, `180`
  - max_pairs: `50/100/150` (baseline z0.95/0.08/120/120)
- Статус: `completed` (18 прогонов: 9 holdout + 9 stress).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|
| z0p95/exit0p06/hold120/cd120 | 4.543 | 450.51 | 3936 | 168 | 107.86 | 3.727 | 369.19 | 3936 | 168 | 191.75 |
| z0p95/exit0p10/hold120/cd120 | 4.542 | 450.59 | 3936 | 168 | 107.86 | 3.727 | 369.28 | 3936 | 168 | 191.76 |
| z0p95/exit0p08/hold120/cd120/maxpairs150 | 4.519 | 447.87 | 3936 | 168 | 107.86 | 3.702 | 366.56 | 3936 | 168 | 191.75 |
| z0p95/exit0p08/hold90/cd90 | 4.519 | 447.87 | 3936 | 168 | 107.86 | 3.702 | 366.56 | 3936 | 168 | 191.75 |
| z0p95/exit0p08/hold180/cd180 | 4.519 | 447.87 | 3936 | 168 | 107.86 | 3.702 | 366.56 | 3936 | 168 | 191.75 |
| z0p95/exit0p08/hold120/cd120/maxpairs50 | 4.270 | 326.89 | 2269 | 120 | 67.74 | 3.592 | 274.72 | 2269 | 120 | 120.43 |
| z0p95/exit0p08/hold120/cd120/maxpairs100 | 4.315 | 426.38 | 3709 | 163 | 105.54 | 3.514 | 346.85 | 3709 | 163 | 187.63 |
| z1p0/exit0p08/hold120/cd120 | 4.090 | 397.73 | 3102 | 168 | 85.38 | 3.426 | 332.92 | 3102 | 168 | 151.79 |
| z0p9/exit0p08/hold120/cd120 | 4.360 | 394.07 | 5043 | 168 | 136.50 | 3.225 | 290.91 | 5043 | 168 | 242.67 |

Выводы:
- Лучший min‑Sharpe (holdout/stress): z0.95 / exit0.06 / hold120 / cd120 (4.54 / 3.73).
- exit0.10 даёт почти идентичный результат; оба лучше базового exit0.08.
- hold90/hold180 и maxpairs150 совпадают с базой — вероятно, ограничения не активируются.
- maxpairs50 заметно снижает turnover (2269 trades, 120 pairs) при Sharpe 4.27/3.59.

### Очереди статусов
- `coint4/artifacts/wfa/aggregate/20260119_relaxed8_nokpss_u250_turnover_stress/run_queue.csv` переведена в `completed` (соответствующие результаты уже в `coint4/artifacts/wfa/runs/20260120_relaxed8_nokpss_u250_turnover_full`).
