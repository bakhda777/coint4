# AGENTS

Цель: убрать двусмысленность в структуре репозитория и в том, где лежат результаты/журналы.

## Канонические пути
- Git-репозиторий (root): `/home/claudeuser/coint4`
- Рабочее приложение (Poetry/CLI/скрипты): `coint4/` (то есть `/home/claudeuser/coint4/coint4`)
- Документация: `docs/` (в корне репо). Для удобства в app-root есть ссылка `coint4/docs -> ../docs`.

## Единые команды (проверки)
Из корня репозитория:
- `make setup` (требует Poetry; ставит зависимости в `coint4/.venv`)
- `make lint` (минимальный ruff: синтаксис/undefined names)
- `make test` (pytest по умолчанию: `not slow and not serial`, см. `coint4/pytest.ini`)
- `make test-serial`, `make test-slow`
- `make ci` (lint + test)

## Где лежат результаты WFA/оптимизаций
- Артефакты запусков (тяжёлые): `coint4/artifacts/wfa/runs/<run_group>/<run_id>/`
- Очереди (маленькие, держим в Git): `coint4/artifacts/wfa/aggregate/<run_group>/run_queue.csv`
- Rollup индекс (маленький, держим в Git): `coint4/artifacts/wfa/aggregate/rollup/run_index.(csv|json|md)`

## Документация результатов
- Состояние: `docs/optimization_state.md` (1 файл, “что сейчас и что дальше”)
- Журналы: `docs/optimization_runs_YYYYMMDD.md` (дневник прогонов)
- Правило: после каждого блока прогонов обновлять `docs/optimization_state.md` и дописывать дневник.

## Статусы очередей (важно для rollup)
- Если прогоны запускались НЕ через `scripts/optimization/run_wfa_queue.py` или `scripts/optimization/watch_wfa_queue.sh` (например, вручную через `run_wfa_fullcpu.sh`), то в `run_queue.csv` статус остаётся `planned`.
- После ручных запусков синхронизировать статусы и пересобрать rollup:
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/<group>/run_queue.csv`
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`

## Исполнение / серверы
- На этом сервере (146.103.41.248) не запускать тяжёлые прогоны.
- Тяжёлые WFA/оптимизации/долгие бэктесты выполнять на 85.198.90.128 (через remote helper ниже).
- Важно: после завершения прогонов на 85.198.90.128 сервер **выключать** (не оставлять включённым). Предпочитать запуск через `coint4/scripts/remote/run_server_job.sh` (по умолчанию `STOP_AFTER=1`).
- Guardrail: `watch_wfa_queue.sh` требует явный `walk_forward.max_steps` и проверяет `max_steps<=5` для queue-прогонов. Длинные “fullspan” сценарии запускать отдельным пайплайном и явно фиксировать в `docs/`.

## Артефакты и Git
- Не коммитить тяжёлые артефакты из `coint4/artifacts/wfa/runs/` (equity curves, PNG, trade stats и т.п.).
- В репозитории фиксируем: `coint4/configs/`, очереди `coint4/artifacts/wfa/aggregate/`, rollup `coint4/artifacts/wfa/aggregate/rollup/`, и `docs/`.

## Безопасность
- Никаких ключей/токенов в репозитории. `SERVSPACE_API_KEY`, `BYBIT_API_KEY`, `BYBIT_API_SECRET` задавать через env или локальные `.env` (не коммитить).
- Если ключ когда-либо попадал в Git/логи, его нужно перевыпустить у провайдера.

## Remote runs (Serverspace)
- API docs: https://docs.serverspace.ru/public_api.html
- Скрипт: `coint4/scripts/remote/run_server_job.sh`
- Переменные: `SERVSPACE_API_KEY`, `SERVER_ID` (или `SERVER_NAME`), `SERVER_IP` (по умолчанию `85.198.90.128`)
- Опции: `SKIP_POWER=1`, `STOP_AFTER=0/1`, `UPDATE_CODE=1/0`, `SYNC_BACK=1/0`, `SYNC_PATHS`, `SSH_KEY`, `SERVER_REPO_DIR`, `SERVER_WORK_DIR`
- Примечание: `STOP_AFTER=1` (default) = **автоматически выключить VPS** после выполнения команды. Не ставить `STOP_AFTER=0` без явной причины.
- Пример (из `coint4/`):
  - `export SERVSPACE_API_KEY="***"; export SERVER_ID="***"; export SERVER_IP="85.198.90.128"`
  - `bash scripts/remote/run_server_job.sh bash scripts/optimization/watch_wfa_queue.sh --queue artifacts/wfa/aggregate/<group>/run_queue.csv`

## Live trading / cutover
- Paper trading (demo/testnet) **не делаем и не планируем**.
- Прод-конфиг: `configs/prod_final_budget1000.yaml`
- План cutover: запуск **сразу в live** (без paper-этапа). Команда (из `coint4/`): `BYBIT_ENV=live BYBIT_API_KEY="***" BYBIT_API_SECRET="***" PYTHONPATH=src ./.venv/bin/python scripts/run_live.py --config configs/prod_final_budget1000.yaml --env live`
- Мониторинг: логи в `coint4/artifacts/live/logs`, состояние `coint4/artifacts/live/state.json`, снапшоты через `coint4/scripts/extract_live_snapshot.py` (обновляет `coint4/artifacts/live/LIVE_DASHBOARD.md`).

## Язык
- Всегда общаемся на русском языке.
