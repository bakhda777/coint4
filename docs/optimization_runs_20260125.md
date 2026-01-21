# Журнал прогонов оптимизации (2026-01-25)

Назначение: снизить turnover и stress cost_ratio через усиленный churn‑control и ограничение числа пар (top20/top30/top50) при realistic costs.

## Обновления (2026-01-25)

### Rollup и gating
- Rollup обновлен: `coint4/artifacts/wfa/aggregate/rollup/run_index.csv` (всего 498 прогонов, метрики: 386).
- Gating: `total_costs>0`, `total_trades>=500`, `total_pairs_traded>=50`, `sharpe_ratio_abs>0`, `total_pnl>0` → holdout: 70, stress: 74.
- Stress c cost_ratio<=0.5: 12 прогонов (в основном top20/top30‑кап).
- Артефакт shortlist: `coint4/artifacts/wfa/aggregate/20260125_realcost_churngrid/shortlist_gate_20260125.csv`.

### Deep‑dive базового кандидата (top50, hold180/cd180/ms0.2)
- run_id: `holdout_relaxed8_nokpss_20260123_..._top50_z1p00_exit0p06_hold180_cd180_ms0p2`.
- Концентрация PnL (по парам, агрегировано): top10 ≈ 58% (holdout), ≈ 66% (stress).
- Негативные пары: 39/120 (holdout), 40/120 (stress).
- Stress cost_ratio ~0.60 → нужен более жёсткий churn‑control и/или cap пар.

### Новый churn‑grid (realcost_churngrid)
- Цель: снизить turnover/cost_ratio через:
  - min_hold/cooldown 240m,
  - min_spread_move_sigma 0.2/0.3,
  - max_pairs 20/30/50.
- Конфиги:
  - `coint4/configs/holdout_20260125_relaxed8_nokpss_u250_churnfix_costgrid/*.yaml` (6 шт.)
  - `coint4/configs/stress_20260125_relaxed8_nokpss_u250_churnfix_costgrid/*.yaml` (6 шт.)
- Очередь: `coint4/artifacts/wfa/aggregate/20260125_realcost_churngrid/run_queue.csv` (12 прогонов, статус: `completed`).
- Запуск только на 85.198.90.128:
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py --queue artifacts/wfa/aggregate/20260125_realcost_churngrid/run_queue.csv --parallel $(nproc)`

#### Результаты (holdout + stress, real costs)
| run_id | kind | sharpe | pnl | costs | trades | pairs | max_dd | cost_ratio |
|---|---|---|---|---|---|---|---|---|
| top20/hold180/cd180/ms0.2 | holdout | 8.12 | 954.8 | 132.2 | 4704 | 53 | -47.1 | 0.14 |
| top20/hold240/cd240/ms0.2 | holdout | 6.70 | 781.0 | 118.2 | 4240 | 53 | -52.5 | 0.15 |
| top30/hold180/cd180/ms0.2 | holdout | 8.79 | 1151.4 | 189.5 | 6865 | 75 | -60.7 | 0.16 |
| top30/hold240/cd240/ms0.2 | holdout | 6.41 | 846.5 | 169.8 | 6196 | 75 | -61.3 | 0.20 |
| top50/hold240/cd240/ms0.2 | holdout | 5.49 | 795.6 | 292.4 | 10223 | 120 | -87.9 | 0.37 |
| top50/hold240/cd240/ms0.3 | holdout | 5.86 | 853.3 | 291.9 | 10196 | 120 | -84.1 | 0.34 |
| top20/hold180/cd180/ms0.2 | stress | 7.25 | 852.5 | 235.0 | 4704 | 53 | -48.8 | 0.28 |
| top20/hold240/cd240/ms0.2 | stress | 5.91 | 689.3 | 210.1 | 4240 | 53 | -67.2 | 0.30 |
| top30/hold180/cd180/ms0.2 | stress | 7.75 | 1015.8 | 337.0 | 6865 | 75 | -63.4 | 0.33 |
| top30/hold240/cd240/ms0.2 | stress | 5.48 | 724.2 | 301.8 | 6196 | 75 | -61.7 | 0.42 |
| top50/hold240/cd240/ms0.2 | stress | 4.39 | 637.0 | 519.9 | 10223 | 120 | -93.1 | 0.82 |
| top50/hold240/cd240/ms0.3 | stress | 4.77 | 694.8 | 519.0 | 10196 | 120 | -90.1 | 0.75 |

#### Выводы
- top20/top30 существенно снизили cost_ratio (0.28–0.33 в stress) при хорошем Sharpe; top50 остаётся дорогим.
- Лучший компромисс: top30/hold180/cd180/ms0.2 (Sharpe 7.75 stress, cost_ratio 0.33, trades 6865).
- hold240 ухудшает Sharpe и не даёт заметного выигрыша по cost_ratio.

### Следующие шаги
- Зафиксировать shortlist: top30/hold180/cd180/ms0.2 и топ20/hold180/cd180/ms0.2 как кандидаты с низким cost_ratio.
- Повторить WFA на альтернативном OOS периоде для top30/top20 (holdout+stress).
- Проверить устойчивость на более строгом funding/slippage стрессе и подтвердить стабильность Sharpe.
