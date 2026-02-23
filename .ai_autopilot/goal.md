# Sharpe>3 autopilot — долгосрочная цель

## North Star (машино-проверяемо)
Достичь **устойчивого** качества по OOS/rollup метрикам (rollup `run_index.*`) так, чтобы лучший кандидат проходил hard-gates:
- `Sharpe >= 3.0` (поле `sharpe` в rollup),
- `|DD| <= 0.30` (абсолют от `max_drawdown_on_equity` / `dd_abs`),
- `trades >= 1000` (поле `trades`, защита от “2 сделки и Sharpe 10”).

Примечание: это пороги **авто-стопа** менеджера (см. ниже). “Устойчивость” трактуем консервативно: после достижения hard-gates дальнейшие спринты должны показывать плато (нет существенного улучшения best Sharpe).

## Ограничения и guardrails
- **На этом сервере тяжёлые прогоны НЕ запускать.** WFA/оптимизации/долгие бэктесты — только на `85.198.90.128` через `coint4/scripts/remote/run_server_job.sh` (по умолчанию `STOP_AFTER=1`).
- Не коммитить тяжёлые артефакты из `coint4/artifacts/wfa/runs/`.
- В Git фиксируем “маленькие” артефакты: очереди `coint4/artifacts/wfa/aggregate/**/run_queue.csv` и rollup `coint4/artifacts/wfa/aggregate/rollup/run_index.(csv|json|md)`, плюс `docs/`.
- Никаких ключей/токенов в репозитории и в логах (заголовки запросов не логировать).

## Канонические источники метрик (для менеджера/ретро)
- Rollup индекс: `coint4/artifacts/wfa/aggregate/rollup/run_index.json` (и `.csv/.md`)
- Очереди прогонов: `coint4/artifacts/wfa/aggregate/<run_group>/run_queue.csv`
- Дневник прогонов: `docs/optimization_runs_YYYYMMDD.md`
- Текущее состояние: `docs/optimization_state.md`

## Авто-остановка (как сейчас делает manager)
Менеджер (`tools/sprint_manager.py:_should_stop`) выставляет `done=true` в `.ai_autopilot/state.json` только если:
1) `sprint >= 5` (консервативный guardrail),
2) hard-gates выполнены: `Sharpe>=3.0`, `|DD|<=0.30`, `trades>=1000`,
3) есть плато: за последние 3 завершённых спринта разброс best Sharpe `< 0.01`.

Ручной override: можно поставить `force_done=true` в `.ai_autopilot/state.json`.

## Чеклист “можно остановиться” (human, усиление)
Даже если авто-стоп сработал, перед реальной остановкой (и тем более перед live-cutover) желательно:
1) иметь независимые подтверждения (разные `run_group`/holdout/tailguard),
2) проверить стабильность хвостов/fees/slippage и отсутствие “одного режима”,
3) убедиться, что DD/tails под контролем и нет деградации по периодам.
