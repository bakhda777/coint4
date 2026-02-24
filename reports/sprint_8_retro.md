# Sprint 8 — Retro

- Дата: 2026-02-24
- Epic: ralph-tui-960f7dbc

## Что было целью спринта
Стабилизировать качество и подготовить следующий информативный блок прогонов

## Что сделали (фактически)
- [ralph-tui-1f1fab66] S8: Обновить optimization_state.md по текущему rollup
- [ralph-tui-4247d117] S8: Синхронизировать статусы run_queue и пересобрать rollup (если надо)
- [ralph-tui-4f43db8f] S8: Подготовить remote-очередь tailguard/holdout на топ-кандидатов
- [ralph-tui-a3d148e2] S8: Улучшить критерий отбора кандидатов (DD/tail/trades) в одном месте
- [ralph-tui-22188af7] (pad) Вспомогательная задача #6
- [ralph-tui-1acc1845] S8: Sprint Retro + Plan Next Sprint


## Что прогнали / что проверили
- Парсер собрал типовые метрики (run_index.*) и статусы beads задач по label
- Тяжёлые прогоны на этом сервере не выполнялись


## Метрики (best-effort, из артефактов)
| Sharpe | |DD| | Trades | Run group | Config |
|---:|---:|---:|---|---|
| 8.649 | 0.013 | 3666 | 20260123_relaxed8_nokpss_u250_churnfix_alt_top50_sens | configs/holdout_20260123_relaxed8_nokpss_u250_churnfix_alt_top50_sens/holdout_relaxed8_nokpss_20260123_alt20220901_20230430_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1.yaml |
| 8.599 | 0.015 | 3977 | 20260123_relaxed8_nokpss_u250_churnfix_alt_top50_sens | configs/holdout_20260123_relaxed8_nokpss_u250_churnfix_alt_top50_sens/holdout_relaxed8_nokpss_20260123_alt20220901_20230430_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z0p95_exit0p08_hold180_cd180_ms0p1.yaml |
| 8.539 | 0.013 | 3651 | 20260126_realcost_alt_oos | configs/holdout_20260126_relaxed8_nokpss_u250_churnfix_alt/holdout_relaxed8_nokpss_20260126_alt20220901_20230430_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2.yaml |
| 8.174 | 0.013 | 3519 | 20260126_realcost_alt_oos | configs/holdout_20260126_relaxed8_nokpss_u250_churnfix_alt/holdout_relaxed8_nokpss_20260126_alt20220901_20230430_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top20_z1p00_exit0p06_hold180_cd180_ms0p2.yaml |
| 7.959 | 0.015 | 3836 | 20260122_relaxed8_nokpss_u250_churnfix_alt | configs/holdout_20260122_relaxed8_nokpss_u250_churnfix_alt/holdout_relaxed8_nokpss_20260122_alt20220901_20230430_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_full_z0p95_exit0p06_hold180_cd180_ms0p1.yaml |


## Артефакты и ссылки на файлы
- /home/claudeuser/coint4/coint4/artifacts/wfa/aggregate/rollup/run_index.json
- /home/claudeuser/coint4/coint4/artifacts/wfa/aggregate/rollup/run_index.md
- /home/claudeuser/coint4/coint4/artifacts/wfa/aggregate/rollup/run_index.csv


## Выводы
- Текущий best-effort baseline зафиксирован из rollup.
- Следующий шаг — дешёвые проверки доверия к метрикам + план remote-прогонов.


## Решения (decision log)
Предыдущие:
- S4: retro generated; next sprint planned
- S5: retro generated; next sprint planned
- S5: retro generated; next sprint planned
- S6: retro generated; next sprint planned
- S7: retro generated; next sprint planned
Новые:
- S8: зафиксировать результаты и запланировать следующий спринт


## Следующие гипотезы / следующий спринт
- S9: Обновить optimization_state.md по текущему rollup
- S9: Подготовить remote-очередь tailguard/holdout на топ-кандидатов
- S9: Синхронизировать статусы run_queue и пересобрать rollup (если надо)
- S9: Улучшить критерий отбора кандидатов (DD/tail/trades) в одном месте
- S9: Sprint Retro + Plan Next Sprint
- (pad) Вспомогательная задача #6


## Что не нашли / предупреждения парсера
- candidate gate applied: passed=3163/6719 (min_trades=200, max_dd_abs=0.15, tail_bucket_pnl>=-200)
- task generator produced only 5 tasks; padding to 6


