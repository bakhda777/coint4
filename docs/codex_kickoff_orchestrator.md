# Orchestrator Kickoff (Run This)

Ты начинаешь итерацию оптимизации для coint4.

## Цель
Подобрать параметры для `budget=1000` с прицелом на текущий bridge11 objective:
`worst_robust_sharpe - dd_penalty`.
Учитывать гейты (по данным проекта) и не менять их без отдельного решения.

## Обязательное правило источника истины
- Перед любыми решениями открыть и опираться на `docs/project_context.md`.
- При сравнении прогонов использовать `coint4/artifacts/wfa/aggregate/rollup/run_index.csv` и `canonical_metrics.json`.

## Старт
1) Прочитать:
   - `docs/project_context.md`
   - `coint4/configs/autopilot/budget1000_batch_loop_bridge11_20260217.yaml`
2) Подтвердить baseline config и стартовые `run_group`/очереди.

## Спавн агентов
Последовательно запросить в одном окне:
- `research`
- `backtest`
- `risk`
- `ops`
- `reviewer`

## Итерационный цикл
Выполнять итерации строго в порядке:

1. `baseline`
2. `batch`
3. `ranking`
4. `decision memo`
5. `stop criteria`

### Что должно включать batch
- Запуск очереди через:
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue_powered.py --queue artifacts/wfa/aggregate/<group>/run_queue.csv --postprocess`
- После прогона обновить `artifacts/wfa/aggregate/rollup/run_index.csv`.
- Не запускать изменение windows/metrik вручную внутри активной итерации.

### Что обязательно на каждом завершении итерации
- Сформировать `decision memo` с полями:
  - `what changed` (что менялось по параметрам, если было)
  - `what ran` (команды/очереди/run_group)
  - `what won` (winner + key metrics)
  - `why` (обоснование через objective + gates)
  - `next` (следующие действия)
  - `stop-check` (статус критериев остановки)

## Жёсткое ограничение
Никаких изменений метрик, расчётов objective или окон WFA без отдельного решения и явного фикса приоритета в `decision memo`.
