# Журнал прогонов оптимизации (2026-01-28)

Назначение: mini‑grid по z‑entry/exit для top30/hold180/cd180/ms0.2 (realistic costs).

## Очередь: realcost_zgrid_top30
- Очередь: `coint4/artifacts/wfa/aggregate/20260128_realcost_zgrid_top30/run_queue.csv` (8 прогонов).
- Конфиги:
  - `coint4/configs/holdout_20260128_relaxed8_nokpss_u250_churnfix_zgrid/*.yaml` (4 шт.)
  - `coint4/configs/stress_20260128_relaxed8_nokpss_u250_churnfix_zgrid/*.yaml` (4 шт.)
- Статус: `completed` (первый запуск дал stalled из‑за несоответствия имени файла; повторный запуск завершён).

### Результаты (holdout + stress)
| config | kind | sharpe | pnl | costs | trades | pairs | max_dd | cost_ratio |
|---|---|---|---|---|---|---|---|---|
| z0.95/exit0.06 | holdout | 8.11 | 1038.7 | 195.0 | 7082 | 75 | -64.1 | 0.19 |
| z0.95/exit0.08 | holdout | 7.26 | 938.3 | 202.8 | 7401 | 75 | -58.6 | 0.22 |
| z1.00/exit0.06 | holdout | 8.79 | 1151.4 | 189.5 | 6865 | 75 | -60.7 | 0.16 |
| z1.00/exit0.08 | holdout | 8.20 | 1071.9 | 197.0 | 7151 | 75 | -61.2 | 0.18 |
| z0.95/exit0.06 | stress | 7.04 | 901.6 | 346.6 | 7082 | 75 | -69.5 | 0.38 |
| z0.95/exit0.08 | stress | 6.14 | 793.8 | 360.6 | 7401 | 75 | -66.1 | 0.45 |
| z1.00/exit0.06 | stress | 7.75 | 1015.8 | 337.0 | 6865 | 75 | -63.4 | 0.33 |
| z1.00/exit0.08 | stress | 7.10 | 928.8 | 350.2 | 7151 | 75 | -75.3 | 0.38 |

## Выводы
- Лучший вариант: z1.00/exit0.06 (макс. Sharpe и минимальный cost_ratio среди top30).
- z0.95/exit0.08 самый слабый по Sharpe и cost_ratio.

## Концентрация PnL (top30 z1.00/exit0.06, holdout)
- Доля top10 пар: ~0.71; top20 пар: ~0.96.
- Отрицательных пар: 28 из 75.
- Топ‑базовые активы (грубая оценка, PnL 50/50 на пару): BTCDAI ~8%, BTC ~8%, KASTA ~8%, ETHDAI ~7%, JUV ~7%.
