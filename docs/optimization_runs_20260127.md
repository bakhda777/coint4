# Журнал прогонов оптимизации (2026-01-27)

Назначение: усиленный stress+ для топ‑20/топ‑30 (realistic costs) с повышенными издержками.

## Конфиги stress+
- Базовые кандидаты: top20/top30, hold180/cd180/ms0.2.
- Усиление издержек:
  - `commission_rate_per_leg: 0.0009` (1.5× от stress 0.0006)
  - `slippage_half_spread_multiplier: 3.0` (1.5× от stress 2.0)
  - `commission_pct: 0.0009`, `slippage_pct: 0.0015`, `slippage_stress_multiplier: 2.5`
- Funding: множитель не параметризован в конфиге; используется базовый расчёт funding из данных.

## Очередь: realcost_stressplus
- Очередь: `coint4/artifacts/wfa/aggregate/20260127_realcost_stressplus/run_queue.csv` (2 прогона).
- Конфиги: `coint4/configs/stress_20260127_relaxed8_nokpss_u250_churnfix_plus/*.yaml` (2 шт.).
- Статус: `completed`.

### Результаты (stress+)
| config | sharpe | pnl | costs | trades | pairs | max_dd | cost_ratio |
|---|---|---|---|---|---|---|---|
| top20/hold180/cd180/ms0.2 | 6.25 | 735.5 | 352.5 | 4704 | 53 | -60.9 | 0.48 |
| top30/hold180/cd180/ms0.2 | 6.57 | 860.7 | 505.4 | 6865 | 75 | -72.3 | 0.59 |

## Выводы
- top20 проходит stress+ порог `cost_ratio <= 0.5` (0.48) при Sharpe 6.25.
- top30 не проходит stress+ из‑за cost_ratio 0.59.
