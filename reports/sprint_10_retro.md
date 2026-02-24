# Sprint 10 — Retro

- Дата: 2026-02-24
- Epic: ralph-tui-960f7dbc

## Что было целью спринта
Стабилизировать качество и подготовить следующий информативный блок прогонов

## Что сделали (фактически)
- [ralph-tui-e7c6ba1a] S10: Goal step: приблизиться к Sharpe>3 через очередь 20260223_tailguard_r07_fullspan_confirm_top3
- [ralph-tui-bb87d444] S10: 20260223_tailguard_r07_fullspan_confirm_top3 mini-rollup + robust ranking (coverage-gated)
- [ralph-tui-4fc2cd1e] S10: Выводы по 20260223_tailguard_r07_fullspan_confirm_top3 и следующий шаг к Sharpe>3
- [ralph-tui-d3b59c98] S10: Обновить optimization_state.md (только после mini-rollup)
- [ralph-tui-fd1e37c7] (pad) Вспомогательная задача #6
- [ralph-tui-fc06ce18] S10: Sprint Retro + Plan Next Sprint


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
- S5: retro generated; next sprint planned
- S6: retro generated; next sprint planned
- S7: retro generated; next sprint planned
- S8: retro generated; next sprint planned
- S9: retro generated; next sprint planned
Новые:
- S10: зафиксировать результаты и запланировать следующий спринт


## Следующие гипотезы / следующий спринт
- S11: Goal step: приблизиться к Sharpe>3 через очередь 20260223_tailguard_r07_fullspan_confirm_top3
- S11: 20260223_tailguard_r07_fullspan_confirm_top3 mini-rollup + robust ranking (coverage-gated)
- S11: Выводы по 20260223_tailguard_r07_fullspan_confirm_top3 и следующий шаг к Sharpe>3
- S11: Обновить optimization_state.md (только после mini-rollup)
- S11: Sprint Retro + Plan Next Sprint
- (pad) Вспомогательная задача #6


## Что не нашли / предупреждения парсера
- candidate gate applied: passed=3193/6755 (min_trades=200, max_dd_abs=0.15, tail_bucket_pnl>=-200)
- task generator produced only 5 tasks; padding to 6


