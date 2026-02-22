# План выполнения (DoD) — 2026-02-22

Цель: формализовать критерии завершения для 12 пунктов плана и зафиксировать фактическое выполнение.

## Критерии завершения по пунктам

| # | Пункт плана | Критерий завершения (DoD) |
|---:|---|---|
| 1 | Зафиксировать baseline-решение как неизменное до strict-pass | В `docs/optimization_state.md` и дневнике явно указано: baseline winner не меняется, пока strict `fullspan_v1` не дал pass. |
| 2 | Разделить рабочее дерево по зонам | Есть явный список целевых файлов для интеграции (code/tests/docs) и список шумовых/генерируемых зон, которые не включаются в текущий пакет. |
| 3 | Выполнить канонический postprocess-цикл | Отработали команды `sync_queue_status -> build_run_index --no-auto-sync-status -> strict rank -> diagnostic rank`; результаты зафиксированы в документации. |
| 4 | Свести `run_index` к единому source of truth | В state/дневнике зафиксирован текущий локальный размер `run_index`, дата и причина возможных расхождений с VPS-снимком. |
| 5 | Подготовить атомарный git-пакет | В staging только целевые файлы текущего изменения, без тяжёлых артефактов и посторонних директорий. |
| 6 | Прогнать контроль качества | `make lint` и целевые pytest завершаются успешно; результат указан в дневнике. |
| 7 | Явно развести promote/research профили | В оркестраторе и документации явно различаются strict promote profile и diagnostic research profile с разными `tail_worst_gate_pct`. |
| 8 | Задокументировать DSR-статус и план возврата гейта | В документации зафиксирован временный статус `min_dsr: null` и условия возврата DSR-гейта. |
| 9 | Подготовить следующий экспериментный блок | Сформирован конкретный run_group/очередь и гипотеза, направленная на снижение `worst_step_pnl`. |
| 10 | Зафиксировать безопасный remote-run шаблон | В документации/чек-листе есть команда запуска через `run_server_job.sh` с `SYNC_UP=1`, `STOP_AFTER=1`, и явной проверкой shutdown. |
| 11 | Унифицировать формат отчётности | Дневник и state обновлены по одному шаблону: `run_group`, `queue_path`, `policy`, `selection_mode`, `promotion_verdict`, `rejection_reason`, `postprocess`. |
| 12 | Гейт на promote в live | В документации явно прописан fail-closed rule: без strict-pass promote в live запрещён. |

## Технические ограничения выполнения

- Репозиторий в сильно грязном состоянии (`staged/unstaged/untracked`), поэтому интеграция ведётся только по целевому подмножеству файлов.
- Heavy WFA не запускается локально; только postprocess/rollup/ranking и тесты.
- Для remote heavy-run сохраняется guardrail: `85.198.90.128` + `STOP_AFTER=1`.

## Разделение scope (пункт 2)

Включаем в текущий пакет (code/tests/docs):

- `coint4/scripts/optimization/build_run_index.py`
- `coint4/scripts/optimization/run_wfa_queue.py`
- `coint4/scripts/optimization/run_fullspan_decision_cycle.py`
- `coint4/tests/scripts/test_build_run_index_auto_sync.py`
- `coint4/tests/scripts/test_run_wfa_queue_postprocess.py`
- `coint4/artifacts/wfa/aggregate/rollup/run_index.csv`
- `coint4/artifacts/wfa/aggregate/rollup/run_index.json`
- `coint4/artifacts/wfa/aggregate/rollup/run_index.md`
- `docs/optimization_state.md`
- `docs/optimization_runs_20260222.md`
- `docs/plan_execution_20260222.md`

Не включаем в текущий пакет (шум/генерация/вне scope):

- массовые untracked `coint4/configs/` (сгенерированные YAML);
- массовые untracked `coint4/artifacts/` и прочие heavy-артефакты;
- временные/локальные результаты `papers/`, `tasks/`, экспериментальные docs вне текущего блока.

## Единый шаблон записи блока (пункт 11)

Для каждого нового блока в `docs/optimization_runs_YYYYMMDD.md` и `docs/optimization_state.md` фиксируем:

- `run_group`
- `queue_path`
- `selection_policy`
- `selection_mode`
- `promotion_verdict`
- `rejection_reason`
- `postprocess` (yes/no + команда)
- `run_index entries` (локальный source of truth)

## Promote gate (пункт 12)

- Fail-closed правило: без `strict_profile_rc=0` по `fullspan_v1` и без прохождения hard-gate (`worst_step_pnl >= -0.20 * capital`, `worst_robust_pnl >= 0`, `worst_dd_pct <= 0.50`) обновление live winner запрещено.

## Журнал выполнения

Заполняется по мере выполнения шагов:

- [x] Пункт 1
- [x] Пункт 2
- [x] Пункт 3
- [x] Пункт 4
- [x] Пункт 5
- [x] Пункт 6
- [x] Пункт 7
- [x] Пункт 8
- [x] Пункт 9
- [x] Пункт 10
- [x] Пункт 11
- [x] Пункт 12

Evidence (2026-02-22):

- P1: baseline-lock явно закреплён в `docs/optimization_state.md` и `docs/optimization_runs_20260222.md`.
- P2: scope split зафиксирован в этом файле (включаемые/исключаемые зоны).
- P3: выполнен `run_fullspan_decision_cycle.py` для `20260220_confirm_top10_bl11` и `20260220_top3_fullspan_wfa`; strict `rc=1`, research `rc=0`.
- P4: source of truth закреплён как локальный `run_index.csv` (`8077` записей, snapshot `2026-02-22T17:42:16Z`).
- P5: staged целевой набор добавлен; в индексе также присутствуют ранее staged файлы, оставленные без изменений.
- P6: `make lint` -> `All checks passed!`; targeted pytest -> `28 passed, 2 deselected`.
- P7: в `coint4/scripts/optimization/run_fullspan_decision_cycle.py` добавлены явные profile-логи и `--research-top`, плюс валидация gate-порогов.
- P8: в `docs/optimization_state.md` и `docs/optimization_runs_20260222.md` добавлен план возврата DSR-gate.
- P9: сформированы `coint4/configs/budget1000_autopilot/20260222_tailguard_r01/*.yaml` и очередь `coint4/artifacts/wfa/aggregate/20260222_tailguard_r01/run_queue.csv`.
- P10: safe remote-run шаблон с `SYNC_UP=1`, `STOP_AFTER=1` и проверкой shutdown добавлен в `docs/optimization_runs_20260222.md`.
- P11: единый шаблон полей отчётности закреплён в этом файле, применён в обновлениях `state/runs`.
- P12: fail-closed promote gate зафиксирован в этом файле и подтверждён в `state/runs`.
