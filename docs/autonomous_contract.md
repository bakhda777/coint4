# Autonomous Contract (strict fullspan, fail-closed)

Обновлено: 2026-03-05.

## 1) Scope и source of truth
- Очереди: `coint4/artifacts/wfa/aggregate/<run_group>/run_queue.csv`
- Rollup: `coint4/artifacts/wfa/aggregate/rollup/run_index.csv`
- Runtime state: `coint4/artifacts/wfa/aggregate/.autonomous/fullspan_decision_state.json`
- Decision notes: `coint4/artifacts/wfa/aggregate/.autonomous/decision_notes.jsonl`

## 2) Queue lifecycle statuses

### 2.1 Row status (`run_queue.csv`)
Допустимые значения:
- `planned`
- `running`
- `stalled`
- `failed`
- `error`
- `completed`
- `skipped`
- `active` (legacy alias; при sync трактуется как pending)

Канонические переходы:
- `planned -> running -> completed|stalled`
- `running -> stalled` (stale-running watchdog)
- `stalled|failed|error -> running` (retry/repair)
- `planned|running|stalled|active -> completed` (через `sync_queue_status.py`, если найден `strategy_metrics.csv`)
- `planned|stalled|failed|error -> skipped` (prefilter/quarantine/early-stop)

`completed` и `skipped` считаются terminal для конкретной строки очереди.

### 2.2 Queue quarantine (orphan)
- Файл: `coint4/artifacts/wfa/aggregate/.autonomous/orphan_queues.csv`
- Схема: `queue,until_ts,reason`
- Поведение: queue исключается из выбора до `until_ts`, затем может быть выбрана снова.

## 3) Decision notes schema (jsonl)

Файл: `coint4/artifacts/wfa/aggregate/.autonomous/decision_notes.jsonl`.
Одна строка = один JSON-объект.

Обязательные поля:
- `ts` (string, UTC RFC3339, формат `YYYY-MM-DDTHH:MM:SSZ`)
- `queue` (string, путь к queue или `global`)
- `action` (string, код действия)
- `reason` (string, краткая причина)
- `next_step` (string, следующий шаг)

Минимальный пример:
```json
{"ts":"2026-03-05T08:12:17Z","queue":"artifacts/wfa/aggregate/<group>/run_queue.csv","action":"REJECT","reason":"gate_status=HARD_FAIL reason=no_progress_streak","next_step":"skip_and_select_next_candidate"}
```

## 4) Инварианты strict fullspan
- `selection_policy` фиксирован: `fullspan_v1`.
- `selection_mode` фиксирован: `fullspan`.
- `promote_profile` обязателен для final verdict; `research_profile` только диагностический.
- Primary ranking key: `score_fullspan_v1`.
- `avg_robust_sharpe` только diagnostic key; не используется как primary ranking key.
- Без strict evidence решение всегда `FAIL_CLOSED` (`cutover_permission=FAIL_CLOSED`, `cutover_ready=false`).
- Только `promotion_gatekeeper_agent.py` имеет право выставлять `PROMOTE_ELIGIBLE` и `ALLOW_PROMOTE`.
- `contract_auditor_agent.py` имеет право только понижать решение до fail-closed при несоответствии контракта.

## 5) Promote/cutover contract

Промоут в cutover разрешён только если одновременно выполнены все условия:
1. `strict_pass_count > 0`.
2. `strict_run_group_count >= 2` (два независимых `run_group`).
3. `confirm_count >= 2`.
4. `confirm_verified_lineage_count >= 2` (подтверждённые confirm/fullspan replay по lineage).
5. `strict_gate_status != FULLSPAN_PREFILTER_REJECT`.
6. `contract_hard_pass = true` по `strict_fullspan_holdout_stress_v1`.

Только в этом случае:
- `promotion_verdict = PROMOTE_ELIGIBLE`
- `cutover_permission = ALLOW_PROMOTE`
- `cutover_ready = true`

Иначе:
- strict pass есть, но `<2` run_group: `PROMOTE_DEFER_CONFIRM`.
- run_group достаточно, но confirm/replay недостаточно: `PROMOTE_PENDING_CONFIRM`.
- любой hard-fail: `REJECT` + `FAIL_CLOSED`.
