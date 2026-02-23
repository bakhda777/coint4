# Sharpe>3 autopilot — долгосрочная цель

## North Star
Достичь **устойчивого** результата **Sharpe > 3** (по OOS/rollup метрикам проекта), при контроле:
- просадки (Max Drawdown / DD),
- числа сделок (достаточная статистика, не “2 сделки и Sharpe 10”),
- хвостовых потерь (tail risk) и стабильности по периодам.

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

## Критерии “можно остановиться” (консервативно)
Остановка допускается только если:
1) есть несколько независимых подтверждений (например, разные `run_group`/holdout/tailguard), и
2) улучшения в последние спринты не дают прироста качества (плато),
3) риски (DD/tails) остаются под контролем.

