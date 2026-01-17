# Журнал прогонов оптимизации (2026-01-17)

Назначение: smoke WFA для проверки логирования команд.

## Статусы
- `active` — идет выполнение.
- `candidate` — выбран для валидации.
- `rejected` — отклонен по результатам валидации.
- `aborted` — прерван вручную/по ошибке.
- `legacy/archived` — устаревший или остановленный прогон.

## Обновления (2026-01-17)

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
