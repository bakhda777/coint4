# Sprint 2 — Retro

- Дата: 2026-02-23
- Epic: ralph-tui-960f7dbc

## Что было целью спринта
Стабилизировать качество и подготовить следующий информативный блок прогонов

## Что сделали (фактически)
- [ralph-tui-aefb4673] S2: Проверить ‘слишком высокий Sharpe’ на консистентность метрик
- [ralph-tui-c5b5bd98] S2: Обновить optimization_state.md по текущему rollup
- [ralph-tui-ae66e37d] S2: Синхронизировать статусы run_queue и пересобрать rollup (если надо)
- [ralph-tui-95bf506e] S2: Подготовить remote-очередь tailguard/holdout на топ-кандидатов
- [ralph-tui-b47269b0] S2: Улучшить критерий отбора кандидатов (DD/tail/trades) в одном месте
- [ralph-tui-7a5a102e] S2: Sprint Retro + Plan Next Sprint


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
- S1: retro generated, but sprint not complete; no planning
- S1: retro generated; next sprint planned
Новые:
- S2: зафиксировать результаты и запланировать следующий спринт


## Следующие гипотезы / следующий спринт
- S3: Обновить optimization_state.md по текущему rollup
- S3: Проверить ‘слишком высокий Sharpe’ на консистентность метрик
- S3: Подготовить remote-очередь tailguard/holdout на топ-кандидатов
- S3: Синхронизировать статусы run_queue и пересобрать rollup (если надо)
- S3: Улучшить критерий отбора кандидатов (DD/tail/trades) в одном месте
- S3: Sprint Retro + Plan Next Sprint


## Что не нашли / предупреждения парсера
- (нет)


