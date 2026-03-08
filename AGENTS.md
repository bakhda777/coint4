Break the task into independent subtasks.
Spawn separate agents for each subtask.
Run them in parallel and merge the results.
Для `[explorer]` агентов использовать модель `gpt-5.3-Codex-Spark`.
Синхронизируй наш код с GitHub и мощным VPS настолько, насколько это нужно для корректного запуска прогонов; полное состояние `1 в 1` не требуется.

# AGENTS

Цель: держать в этом файле только стабильные правила для агента и канонические инварианты проекта. Подробные runbook, FAQ и историю изменений хранить в профильных документах из `docs/`.

## Agent Behavior
- Если задача делится на независимые части, сначала делить её на подзадачи, запускать отдельных агентов и сливать результаты.
- `[explorer]` агенты работают в read-only режиме; их каноническая модель: `gpt-5.3-Codex-Spark`.
- Если нет явного стоп-сигнала, агент работает автономно и сам выбирает следующий безопасный шаг.
- Формат отчётности: короткие апдейты по факту, с ключевыми метриками и следующим шагом.
- Всегда общаемся на русском языке.
- Подробный multi-agent workflow: `docs/agents.md`.

## Repo Canon
- Git-репозиторий (root): `/home/claudeuser/coint4`
- Рабочее приложение (Poetry/CLI/скрипты): `coint4/` (то есть `/home/claudeuser/coint4/coint4`)
- Документация: `docs/` (в корне репо); для удобства в app-root есть ссылка `coint4/docs -> ../docs`

## Ralph TUI Canon
- `ralph tui` с `tracker=json` читает задачи из `.ralph-tui/prd.json`; JSON для трекера должен быть в ralph-формате.
- Если нужен богатый PRD со `constraints`/`definitions`/`steps`, хранить его отдельным файлом вроде `prd_<name>.spec.json` или `prd_<name>.md`, а tracker-JSON держать минимальным.

## Validation Canon
- Из корня репозитория использовать единые команды: `make setup`, `make lint`, `make test`, `make test-serial`, `make test-slow`, `make ci`.

## Execution Canon
- На этом сервере (`146.103.41.248`) не запускать тяжёлые прогоны.
- Тяжёлые WFA/оптимизации/долгие бэктесты выполнять только на `85.198.90.128`.
- `85.198.90.128` платный VPS: включать его только под реальный remote batch/run.
- Вне активного прогона `85.198.90.128` по умолчанию должен быть выключен.
- Во время прогона загружать `85.198.90.128` максимально плотно, без искусственного простаивания между шагами одного batch.
- Предпочитать запуск через `coint4/scripts/remote/run_server_job.sh`; после remote job VPS должен быть выключен, по умолчанию `STOP_AFTER=1`.
- Remote execution order: `verify -> run -> fetch`.
- Если для remote run нужны локальные tracked-файлы, которых ещё нет на VPS через `git pull`, использовать `SYNC_UP=1`.
- `watch_wfa_queue.sh` считать manual/debug entrypoint; для queue-прогонов он требует явный `walk_forward.max_steps`, а длинные fullspan-сценарии вести отдельным пайплайном и фиксировать в `docs/`.
- Подробный execution/runbook: `docs/agent_playbook.md`, `docs/openclaw_guardrails.md`, `docs/autonomous_contract.md`.

## Artifacts And Git Canon
- Тяжёлые run-артефакты хранить в `coint4/artifacts/wfa/runs/<run_group>/<run_id>/`; их не коммитить.
- Git-tracked артефакты и очереди хранить в `coint4/configs/`, `coint4/artifacts/wfa/aggregate/`, `coint4/artifacts/wfa/aggregate/rollup/` и `docs/`.
- Эфемерные autonomous search batches `autonomous_seed_*` и `autonomous_seed_check_*` в `coint4/artifacts/wfa/aggregate/` и `coint4/configs/evolution/` считать runtime-local: по умолчанию их не коммитить и не держать в `git status`, пока batch не отобран для явной публикации/разбора.
- Seeder decision-хвосты для эфемерных batch (`coint4/artifacts/wfa/aggregate/autonomous_queue_seeder/decisions/evo_autonomous_queue_seeder_autonomous_seed_*.json`) считать локальными runtime-артефактами.
- Каноническая опубликованная ветка на GitHub: `main`; завершённые изменения должны доходить до `origin/main`.
- Ветки `feature/*`, `session/*`, `ralph-*` использовать только как временные рабочие; после завершения изменений их нужно fast-forward/merge в локальный `main` и пушить в `origin/main`.

## Queue And Rollup Canon
- Очереди хранить в `coint4/artifacts/wfa/aggregate/<run_group>/run_queue.csv`.
- Rollup-индекс хранить в `coint4/artifacts/wfa/aggregate/rollup/run_index.(csv|json|md)`.
- Если прогоны запускались не через `scripts/optimization/run_wfa_queue.py` или `scripts/optimization/watch_wfa_queue.sh`, статус в `run_queue.csv` может остаться `planned`.
- После ручных запусков обязательно синхронизировать статусы и пересобрать rollup:
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/<group>/run_queue.csv`
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`
- Перед любым перезапуском очереди сначала анализировать полный контекст: `queue.log`, `candidate`/`daemon`-логи, статус VPS и последнюю причину статуса из `strategy_metrics`/`equity_curve`/`errors`.
- Ручные команды `heavy-run`/`run_wfa_queue*` не использовать по умолчанию; запускать очередь через непрерывный оркестратор или powered path.

## Reporting Canon
- Состояние процесса вести в `docs/optimization_state.md`.
- Дневник прогонов вести в `docs/optimization_runs_YYYYMMDD.md`.
- Guardrails и частые runtime/preflight проблемы вести в `docs/openclaw_guardrails.md`.
- После каждого блока прогонов обновлять `docs/optimization_state.md` и дописывать дневник за соответствующую дату.

## Live Canon
- Paper trading не делать и не планировать; cutover выполнять сразу в `live`.
- Прод-конфиг: `configs/prod_final_budget1000.yaml`.
- Подробности live/cutover и production checklist: `docs/production_checklist.md`, `docs/project_context.md`.

## Security Canon
- Никаких ключей и токенов в репозитории.
- `SERVSPACE_API_KEY`, `BYBIT_API_KEY`, `BYBIT_API_SECRET` задавать через env, локальные `.env` или `.secrets/serverspace_api_key`; не коммитить и не печатать их в stdout/stderr/логах.
- Если ключ когда-либо попал в Git, чат или логи, считать его скомпрометированным и перевыпустить у провайдера.
