# Журнал прогонов оптимизации (2026-01-16)

Назначение: продолжение после ночного отключения сервера и возобновление очередей.

## Статусы
- `active` — идет выполнение.
- `candidate` — выбран для валидации.
- `rejected` — отклонен по результатам валидации.
- `aborted` — прерван вручную/по ошибке.
- `legacy/archived` — устаревший или остановленный прогон.

## Состояние после отключения
- Все запуски остановлены в середине WFA; очередь возобновления оформлена через `run_queue.csv` и списки конфигов.
- Возобновление фиксируем в этом журнале, с указанием фильтрации пар по шагам (stdout/run.log).
- Лимит WFA: максимум 5 шагов без отдельного согласования.
- Состояние выполнения фиксируем в `docs/optimization_state.md` (кратко: что сделано и что дальше).

## Обновления (2026-01-16)

### SSD top-N sweep (6 значений, resume)

#### ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd10000
- Метрики (strategy_metrics.csv): total_pnl `489.63`, sharpe_ratio_abs `0.4998`, max_drawdown_abs `-111.53`, total_trades `5898`, total_pairs_traded `249.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_ssd_topn_sweep/ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd10000/`.
- Статус: `completed` (возобновлено после остановки).

Сводка фильтрации пар (из run.log, Q4 2023):
```yaml
step: 1
candidates_total: 10000
passed_pairs: 103
remaining_after_stage:
  after_low_correlation: 9431
  after_beta: 6370
  after_mean_crossings: 6370
  after_half_life: 6320
  after_pvalue: 4730
  after_hurst: 3251
  after_kpss: 103
  after_market_microstructure: 103
---
step: 2
candidates_total: 10000
passed_pairs: 36
remaining_after_stage:
  after_low_correlation: 7803
  after_beta: 5461
  after_mean_crossings: 5461
  after_half_life: 5444
  after_pvalue: 3794
  after_hurst: 3002
  after_kpss: 36
  after_market_microstructure: 36
---
step: 3
candidates_total: 10000
passed_pairs: 118
remaining_after_stage:
  after_low_correlation: 9558
  after_beta: 6637
  after_mean_crossings: 6637
  after_half_life: 6633
  after_pvalue: 4331
  after_hurst: 2807
  after_kpss: 118
  after_market_microstructure: 118
```

#### ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd25000
- Метрики (strategy_metrics.csv): total_pnl `1205.94`, sharpe_ratio_abs `0.6502`, max_drawdown_abs `-199.31`, total_trades `11041`, total_pairs_traded `520.0`.
- Артефакты: `coint4/artifacts/wfa/runs/20260115_ssd_topn_sweep/ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd25000/`.
- Статус: `completed`.
- Примечание: в метриках total_costs = `0.0` — проверить учет комиссий/слиппеджа.

Сводка фильтрации пар (из filter_reasons_20260116_125227/125933/130854.csv, Q4 2023):
```yaml
step: 1
candidates_total: 25000
passed_pairs: 238
remaining_after_stage:
  after_low_correlation: 20216
  after_beta: 13215
  after_mean_crossings: 13215
  after_half_life: 13164
  after_pvalue: 10517
  after_hurst: 7389
  after_kpss: 238
  after_market_microstructure: 238
---
step: 2
candidates_total: 25000
passed_pairs: 104
remaining_after_stage:
  after_low_correlation: 14874
  after_beta: 10035
  after_mean_crossings: 10035
  after_half_life: 10009
  after_pvalue: 7486
  after_hurst: 6030
  after_kpss: 104
  after_market_microstructure: 104
---
step: 3
candidates_total: 25000
passed_pairs: 190
remaining_after_stage:
  after_low_correlation: 22285
  after_beta: 15291
  after_mean_crossings: 15291
  after_half_life: 15258
  after_pvalue: 9573
  after_hurst: 6358
  after_kpss: 190
  after_market_microstructure: 190
```

#### ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd50000
- Статус: `active` (WF-шаг 3/3 выполняется).
- Последний чекпойнт: WF-шаг 2 завершен, после backtests `122` пар, P&L шага `+355.58` (run.log).
- Артефакты: `coint4/artifacts/wfa/runs/20260115_ssd_topn_sweep/ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd50000/`.
- Примечание: strategy_metrics.csv еще не сформирован.

### Signal grid (z=0.75/0.8/0.85/0.9 × exit=0.04/0.06/0.08/0.1)
- 16 конфигов, ssd_top_n `25000`, max_steps `3`.
- Очередь: `coint4/artifacts/wfa/aggregate/20260116_signal_grid/run_queue.csv`.
- Статус: `active` (parallel `16`, n_jobs `1`).
- Артефакты: `coint4/artifacts/wfa/runs/20260116_signal_grid/`.
- Фильтрация пар по шагам WFA — будет добавлена по завершении (run.log/filter_reasons_*.csv).

### Piogoga grid (leader filters, zscore sweep)
- 16 конфигов, z=0.75/0.8/0.85/0.9 × exit=0.05/0.06/0.07/0.08, ssd_top_n `25000`, n_jobs `1`, max_steps `3`.
- База фильтров: pv=0.4, kpss=0.05, hurst<=0.65, corr>=0.4, hl=0.001-100 (лидер SSD sweep).
- Сгенерированные конфиги: `coint4/configs/piogoga_grid_20260116/` (manifest.csv).
- Очередь: `coint4/artifacts/wfa/aggregate/20260116_piogoga_grid/run_queue.csv`.
- Артефакты: `coint4/artifacts/wfa/runs/20260116_piogoga_grid/`.
- Статус: `planned` (parallel `16`, n_jobs `1`).
- Фильтрация пар по шагам WFA — будет добавлена по завершении (run.log/filter_reasons_*.csv).

### Leader validation (post-analysis, SSD leader)
- Основание: лидер rollup по Sharpe/PnL — `ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd25000`.
- Конфиг: `coint4/configs/leader_validation_20260116/leader_validate_20260116_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd25000.yaml`.
- Период WFA: `2023-10-01` → `2024-04-30`, 60/30, max_steps `5`.
- Фильтрация пар: параллельная (берет `backtest.n_jobs`, backend `threads` по умолчанию; можно форсировать `COINT_FILTER_BACKEND=processes`, используется `spawn` для OpenMP‑безопасности).
- Очередь: `coint4/artifacts/wfa/aggregate/20260116_leader_validation/run_queue.csv`.
- Артефакты: `coint4/artifacts/wfa/runs/20260116_leader_validation/`.
- Метрики (strategy_metrics.csv): total_pnl `1388.71`, sharpe_ratio_abs `0.5255`, max_drawdown_abs `-199.31`, total_trades `17096`, total_pairs_traded `775`.
- Примечание: total_costs = `0.0` — проверить учет комиссий/слиппеджа.
- Статус: `completed` (parallel `1`, n_jobs `-1`, backend `processes`).

Сводка фильтрации пар (из run.log, WFA 5 шагов):
```yaml
step: 1
candidates_total: 25000
passed_pairs: 238
remaining_after_stage:
  after_low_correlation: 20216
  after_beta: 13215
  after_mean_crossings: 13215
  after_half_life: 13164
  after_pvalue: 10517
  after_hurst: 7389
  after_kpss: 238
  after_market_microstructure: 238
---
step: 2
candidates_total: 25000
passed_pairs: 104
remaining_after_stage:
  after_low_correlation: 14874
  after_beta: 10035
  after_mean_crossings: 10035
  after_half_life: 10009
  after_pvalue: 7486
  after_hurst: 6030
  after_kpss: 104
  after_market_microstructure: 104
---
step: 3
candidates_total: 25000
passed_pairs: 190
remaining_after_stage:
  after_low_correlation: 22285
  after_beta: 15291
  after_mean_crossings: 15291
  after_half_life: 15258
  after_pvalue: 9573
  after_hurst: 6358
  after_kpss: 190
  after_market_microstructure: 190
---
step: 4
candidates_total: 25000
passed_pairs: 158
remaining_after_stage:
  after_low_correlation: 19205
  after_beta: 13022
  after_mean_crossings: 13022
  after_half_life: 12982
  after_pvalue: 8106
  after_hurst: 5887
  after_kpss: 158
  after_market_microstructure: 158
---
step: 5
candidates_total: 25000
passed_pairs: 128
remaining_after_stage:
  after_low_correlation: 11234
  after_beta: 7916
  after_mean_crossings: 7916
  after_half_life: 7916
  after_pvalue: 7011
  after_hurst: 5426
  after_kpss: 128
  after_market_microstructure: 128
```

### Leader holdout (active)
- Основание: rollup‑лидер по композиту Sharpe/PnL/DD/стабильность — `ssd_topn_20260115_exit0p06_pv0p4_kpss0p05_h0p65_c0p4_hl0p001-100_ssd25000`.
- Конфиг: `coint4/configs/best_config__leader_holdout_ssd25000__20260116_211943.yaml`.
- Окно WFA: `2024-05-01` → `2024-12-31`, 60/30, max_steps `5`.
- Очередь: `coint4/artifacts/wfa/aggregate/20260116_leader_holdout/run_queue.csv`.
- Артефакты: `coint4/artifacts/wfa/runs/20260116_leader_holdout/best_config__leader_holdout_ssd25000__20260116_211943/`.
- Статус: `active` (CPU‑watcher, COINT_FILTER_BACKEND=processes, backtest.n_jobs=-1).
- Фильтрация пар по шагам WFA — будет добавлена после завершения (run.log/filter_reasons_*.csv).

Команда запуска (из `coint4/`):
```bash
COINT_FILTER_BACKEND=processes bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260116_leader_holdout/run_queue.csv \
  --parallel 1
```

## Очереди на возобновление
- SSD top-N sweep (6 значений): `coint4/artifacts/wfa/aggregate/20260115_ssd_topn_sweep/run_queue.csv`.
- SSD top-N sweep (3 значения): `coint4/artifacts/wfa/aggregate/20260115_ssd_topn_sweep_3vals/run_queue.csv`.
- Strict PV grid: `coint4/artifacts/wfa/aggregate/20260115_selgrid_strictpv/run_queue.csv` (resumable очередь).
- Selection grid: `coint4/artifacts/wfa/aggregate/20260115_selgrid/run_queue.csv` (resumable очередь).
- SSD refine (20k/25k/30k/40k): `coint4/artifacts/wfa/aggregate/20260116_ssd_topn_refine/run_queue.csv`.
- Signal sweep (z=0.75/0.8/0.85 × exit=0.04/0.06/0.08): `coint4/artifacts/wfa/aggregate/20260116_signal_sweep/run_queue.csv`.
- Signal grid (z=0.75/0.8/0.85/0.9 × exit=0.04/0.06/0.08/0.1): `coint4/artifacts/wfa/aggregate/20260116_signal_grid/run_queue.csv`.
- Risk sweep (stop/time/cooldown/rolling): `coint4/artifacts/wfa/aggregate/20260116_risk_sweep/run_queue.csv`.
- Piogoga grid (leader filters, zscore sweep): `coint4/artifacts/wfa/aggregate/20260116_piogoga_grid/run_queue.csv`.
- Leader validation (post-analysis): `coint4/artifacts/wfa/aggregate/20260116_leader_validation/run_queue.csv`.

## Команды возобновления (из `coint4/`)

WFA очереди (stalled/planned, через CPU‑watcher):
```bash
bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260115_ssd_topn_sweep/run_queue.csv

bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260115_ssd_topn_sweep_3vals/run_queue.csv
```

WFA очереди с headless Codex по завершении:
```bash
bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260115_ssd_topn_sweep/run_queue.csv \
  --on-done-prompt-file scripts/optimization/on_done_codex_prompt.txt \
  --on-done-log artifacts/wfa/aggregate/20260115_ssd_topn_sweep/codex_on_done.log
```

SSD refine:
```bash
bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260116_ssd_topn_refine/run_queue.csv
```

Signal sweep:
```bash
bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260116_signal_sweep/run_queue.csv
```

Signal grid:
```bash
bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260116_signal_grid/run_queue.csv \
  --parallel 16
```

Piogoga grid:
```bash
bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260116_piogoga_grid/run_queue.csv \
  --parallel 16
```

Leader validation (single run, full CPU):
```bash
bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260116_leader_validation/run_queue.csv
```

Risk sweep:
```bash
bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260116_risk_sweep/run_queue.csv
```

Strict PV grid (queue, `n_jobs: -1` внутри конфигов):
```bash
bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260115_selgrid_strictpv/run_queue.csv
```

Selection grid (queue выбранных конфигов):
```bash
bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260115_selgrid/run_queue.csv
```

## Rollup индекс прогонов
```bash
PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py \
  --output-dir artifacts/wfa/aggregate/rollup
```

## Ссылки
- План дальнейшей оптимизации: `docs/optimization_plan_20260116.md`.
