# Журнал прогонов оптимизации (2026-01-17)

Назначение: smoke WFA для проверки логирования команд.

## Статусы
- `active` — идет выполнение.
- `candidate` — выбран для валидации.
- `rejected` — отклонен по результатам валидации.
- `aborted` — прерван вручную/по ошибке.
- `legacy/archived` — устаревший или остановленный прогон.

## Обновления (2026-01-17)

### WFA очередь (next5_fast, signal/risk sweeps)
- Очередь: `coint4/artifacts/wfa/aggregate/20260117_next5_fast/run_queue_next5_fast.csv`.
- Запуск: `COINT_FILTER_BACKEND=threads ./run_wfa_fullcpu.sh <config> <results_dir>` по одному прогону; очередь используется как список и для ручного статуса (1 running, остальные planned).
- Параллельность: `1` (последовательно), CPU загружается через `backtest.n_jobs=-1` в конфиге.
- Конфиги: `coint4/configs/_tmp_fast_next10/*.yaml` (обновлено `backtest.n_jobs: -1`).
- Артефакты: `coint4/artifacts/wfa/runs/20260117_next5_fast/`.
- Логи: `coint4/artifacts/wfa/runs/20260117_next5_fast/<run>/run.log`.
- Нагрузка CPU (snapshot): coint2 ~130-160% CPU при 8 vCPU (системно ~15-18% user, load avg ~1.2) - есть запас для ускорения внутри одного прогона.
- Прогон 1: `configs/_tmp_fast_next10/signal_sweep_20260116_z0p75_exit0p04_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260116_z0p75_exit0p04_ssd25000` (done in `943.68s`, end `2026-01-17T14:01:27Z`).
- Метрики (strategy_metrics.csv): total_pnl `703.04`, sharpe_ratio_abs `0.4280`, max_drawdown_abs `-256.19`, total_trades `5595`, total_pairs_traded `197`, total_costs `0.0`, win_rate `0.6264`.
- Фильтрация пар (step 1-3): `coint4/results/filter_reasons_20260117_135140.csv`, `coint4/results/filter_reasons_20260117_135600.csv`, `coint4/results/filter_reasons_20260117_140117.csv`.
- Сводка причин отсева (по категориям, rows):
```yaml
step_1:
  total_rows: 24906
  pvalue: 7829
  beta_out_of_range: 6518
  low_correlation: 6063
  hurst_too_high: 2375
  kpss: 2049
  half_life: 72
step_2:
  total_rows: 24954
  low_correlation: 12320
  pvalue: 5447
  beta_out_of_range: 4185
  kpss: 1814
  hurst_too_high: 1140
  half_life: 48
step_3:
  total_rows: 24943
  pvalue: 11764
  beta_out_of_range: 6688
  low_correlation: 3384
  hurst_too_high: 1840
  kpss: 1212
  half_life: 55
```
- Прогон 2: `configs/_tmp_fast_next10/signal_sweep_20260116_z0p75_exit0p08_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260116_z0p75_exit0p08_ssd25000` (status `completed`).
- Метрики (strategy_metrics.csv): total_pnl `707.41`, sharpe_ratio_abs `0.4315`, max_drawdown_abs `-256.19`, total_trades `5598`, total_pairs_traded `197`, win_rate `0.6264`.
- Примечание: попытка с `COINT_FILTER_BACKEND=processes` упала на `PermissionError: [Errno 13] Permission denied` (semlock); перезапуск на threads.
- Legacy/archived: запуск очереди через `COINT_FILTER_BACKEND=threads bash scripts/optimization/watch_wfa_queue.sh --queue artifacts/wfa/aggregate/20260117_next5_fast/run_queue_next5_fast.csv --parallel 1` остановлен, оставлен только для истории.
- Прогон 3: `configs/_tmp_fast_next10/signal_sweep_20260116_z0p8_exit0p06_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260116_z0p8_exit0p06_ssd25000` (done `2026-01-17T16:27:43Z`).
- Метрики (strategy_metrics.csv): total_pnl `771.63`, sharpe_ratio_abs `0.5860`, max_drawdown_abs `-146.72`, total_trades `4334`, total_pairs_traded `197`, win_rate `0.6813`.
- Фильтрация пар (step 1-3): `coint4/results/filter_reasons_20260117_161818.csv`.
- Прогон 4: `configs/_tmp_fast_next10/signal_sweep_20260116_z0p85_exit0p06_ssd25000.yaml` → `coint4/artifacts/wfa/runs/20260117_next5_fast/signal_sweep_20260116_z0p85_exit0p06_ssd25000` (done `2026-01-17T18:05:14Z`).
- Метрики (strategy_metrics.csv): total_pnl `815.67`, sharpe_ratio_abs `0.6345`, max_drawdown_abs `-132.02`, total_trades `3406`, total_pairs_traded `197`, win_rate `0.6593`.
- Фильтрация пар (step 1-3): `coint4/results/filter_reasons_20260117_175527.csv`.
- Статус: `active` (последовательный запуск для анализа между прогонами).

### Smoke WFA (command logging verification)
- Конфиг: `coint4/configs/main_2024_smoke.yaml` (max_steps=1, n_jobs=-1).
- Команда: `COINT_FILTER_BACKEND=processes ./run_wfa_fullcpu.sh configs/main_2024_smoke.yaml artifacts/wfa/runs/logging_smoke_20260117_072821`.
- Артефакты: `coint4/artifacts/wfa/runs/logging_smoke_20260117_072821/`.
- Командный лог: `coint4/artifacts/wfa/runs/logging_smoke_20260117_072821/run.commands.log`.
- Фильтрация пар: `coint4/results/filter_reasons_20260117_072907.csv`.
- Метрики (strategy_metrics.csv): total_pnl `-52.40`, sharpe_ratio_abs `-0.6205`, max_drawdown_abs `-66.69`, total_trades `394`, total_pairs_traded `108`, total_costs `0.0`.
- Статус: `completed`.

Сводка фильтрации пар (из stdout):
```yaml
step: 1
candidates_total: 2000
passed_pairs: 108
remaining_after_stage:
  after_low_correlation: 1302
  after_beta: 920
  after_mean_crossings: 920
  after_half_life: 427
  after_pvalue: 162
  after_hurst: 108
  after_kpss: 108
  after_market_microstructure: 108
```
