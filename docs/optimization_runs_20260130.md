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
- Статус: `completed`.

## Результаты (holdout + stress)
| config | kind | sharpe | pnl | costs | trades | pairs | max_dd | cost_ratio |
|---|---|---|---|---|---|---|---|---|
| top30 z1.00/exit0.06 | holdout | 2.95 | 620.0 | 310.8 | 6384 | 68 | -288.1 | 0.50 |
| top30 z1.00/exit0.06 | stress | 1.87 | 392.6 | 552.5 | 6384 | 68 | -314.1 | 1.41 |
| top20 z1.00/exit0.06 | holdout | 2.44 | 452.6 | 227.3 | 4770 | 55 | -206.9 | 0.50 |
| top20 z1.00/exit0.06 | stress | 1.50 | 276.2 | 404.1 | 4770 | 55 | -252.6 | 1.46 |

## Концентрация PnL (extended OOS, holdout)
- top30: top10 пар ~1.08, top20 пар ~1.42, отрицательных пар 30 из 68; топ-базы: GODS, KCAL, KDA, HFT, FTT, IZI.
- top20: top10 пар ~1.41, top20 пар ~1.69, отрицательных пар 23 из 55; топ-базы: GODS, KCAL, KDA, HFT, FTT, FLOW.

## Выводы
- Extended OOS заметно слабее прежних периодов: Sharpe 1.5-3.0 и стрессовые cost_ratio > 1.0 (издержки превышают PnL).
- Оба кандидата не проходят stress-гейт по издержкам; holdout cost_ratio близок к 0.5, но запас минимальный.
- Концентрация PnL высокая (top10/top20 > 1.0) из-за заметных отрицательных вкладов части пар.

## Рекомендации
- Усилить контроль turnover/издержек для extended OOS (например, рост min_spread_move_sigma или увеличение hold/cooldown) и повторить stress.
- Проверить, не требует ли этот период более жесткого отбора пар (tradeability + funding фильтры) или дополнительного top-k ограничения.
