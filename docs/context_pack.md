# Context Pack: как работать и где что лежит

Источник истины по структуре/правилам: `AGENTS.md` (в корне репо). Здесь только краткая шпаргалка.

## Канонические пути
- Repo root: `/home/claudeuser/coint4`
- App root (Poetry/CLI/скрипты): `coint4/` (то есть `/home/claudeuser/coint4/coint4`)
- Docs: `docs/` (в корне репо), в app-root есть ссылка `coint4/docs -> ../docs`

## Быстрые проверки (запускать из repo root)
- `make setup` (Poetry -> ставит зависимости в `coint4/.venv`)
- `make lint` (ruff: синтаксис/undefined names)
- `make test` (pytest по умолчанию: `not slow and not serial`, см. `coint4/pytest.ini`)
- `make test-serial`, `make test-slow`
- `make ci` (lint + test)

## Где лежат результаты WFA/оптимизаций
- Тяжёлые артефакты прогонов (не коммитим): `coint4/artifacts/wfa/runs/<run_group>/<run_id>/`
- Очереди (маленькие, держим в Git): `coint4/artifacts/wfa/aggregate/<run_group>/run_queue.csv`
- Rollup индекс (маленький, держим в Git): `coint4/artifacts/wfa/aggregate/rollup/run_index.(csv|json|md)`

## Документация прогонов
- Состояние (1 файл): `docs/optimization_state.md`
- Дневники: `docs/optimization_runs_YYYYMMDD.md`
- Правило: после каждого блока прогонов обновлять `docs/optimization_state.md` и дописывать дневник.

## Queue / rollup (если прогоны делались вручную)
Если запускали WFA не через `scripts/optimization/run_wfa_queue.py` или `scripts/optimization/watch_wfa_queue.sh`,
то статусы в `run_queue.csv` останутся `planned`. После ручных прогонов:

```bash
cd coint4
PYTHONPATH=src ./.venv/bin/python scripts/optimization/sync_queue_status.py \
  --queue artifacts/wfa/aggregate/<group>/run_queue.csv
PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py \
  --output-dir artifacts/wfa/aggregate/rollup
```

## Тяжёлые прогоны: только на VPS
- На этом сервере (146.103.41.248) тяжёлые WFA/оптимизации/долгие бэктесты не запускать.
- Тяжёлое исполнять на `85.198.90.128` через `coint4/scripts/remote/run_server_job.sh`.
- По умолчанию `STOP_AFTER=1` выключает VPS по завершении (это желаемое поведение).
- Guardrail: queue-прогоны через `scripts/optimization/watch_wfa_queue.sh` требуют явный `walk_forward.max_steps` и проверяют `max_steps<=5`.
- Принцип remote job: сначала `verify` (проверки), затем `run` (прогоны), затем `fetch` (забрать результаты обратно).

## Git и артефакты
- Не коммитить тяжёлые артефакты из `coint4/artifacts/wfa/runs/**`.
- В Git фиксируем конфиги/очереди/rollup/доки: `coint4/configs/`, `coint4/artifacts/wfa/aggregate/**`, `docs/`.

## Секреты
- Ключи не храним в Git и не печатаем в логи.
- Serverspace API key: только `.secrets/serverspace_api_key` (chmod 600, gitignored) и/или env `SERVSPACE_API_KEY`.
- Bybit: `BYBIT_API_KEY`, `BYBIT_API_SECRET` только через env/локальные `.env` (не коммитить).
