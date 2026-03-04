# AGENTS

Цель: убрать двусмысленность в структуре репозитория и в том, где лежат результаты/журналы.

## Канонические пути
- Git-репозиторий (root): `/home/claudeuser/coint4`
- Рабочее приложение (Poetry/CLI/скрипты): `coint4/` (то есть `/home/claudeuser/coint4/coint4`)
- Документация: `docs/` (в корне репо). Для удобства в app-root есть ссылка `coint4/docs -> ../docs`.

## Ralph TUI: PRD (JSON tracker)
`ralph tui` с `tracker=json` читает PRD из файла JSON (по умолчанию: `.ralph-tui/prd.json`, см. `.ralph-tui/config.toml:trackerOptions.path`). Этот JSON должен быть в **ralph-формате**, иначе UI/трекер не сможет загрузить задачи.

### Канонический формат PRD JSON
Top-level:
- `name` (string) — название эпика/задачи.
- `description` (string, optional) — короткое описание/контекст.
- `branchName` (string) — ветка (например, `feature/<slug>` или `main`).
- `userStories` (array) — список задач, которые показывает `ralph tui`.
- `metadata` (object, optional) — произвольные метаданные; обычно `createdAt/updatedAt/version/sourcePrd`.

`userStories[]`:
- `id` (string) — уникальный ID (например, `C01` или `US-001`).
- `title` (string) — заголовок.
- `description` (string) — короткое описание.
- `acceptanceCriteria` (array[string]) — критерии приёмки (можно `[]`).
- `priority` (int) — чем меньше, тем выше приоритет (обычно 1..5).
- `passes` (bool) — `true` = задача считается выполненной в трекере.
- `labels` (array[string], optional) — теги (можно `[]`).
- `dependsOn` (array[string]) — зависимости по `id` (можно `[]`).
- `completionNotes` (string, optional) — заметка по факту выполнения.

Правило: если нужен “богатый” PRD со `constraints/definitions/steps`, хранить его отдельным файлом (например, `prd_<name>.spec.json` или `prd_<name>.md`), а PRD для `ralph tui` держать в ralph-формате.

Минимальный пример:
```json
{
  "name": "example_epic",
  "description": "Короткий контекст (опционально).",
  "branchName": "feature/example_epic",
  "userStories": [
    {
      "id": "C01",
      "title": "Сделать X",
      "description": "",
      "acceptanceCriteria": [],
      "priority": 2,
      "passes": false,
      "labels": [],
      "dependsOn": []
    }
  ],
  "metadata": {
    "updatedAt": "2026-02-15T00:00:00.000Z"
  }
}
```

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

## Secrets & Infra
- Serverspace Public API key хранить только локально:
  - файл: `.secrets/serverspace_api_key` (chmod 600)
  - env: `SERVSPACE_API_KEY`
- Никогда не коммитить/не печатать ключ в stdout/stderr/логах. В коде/скриптах не логировать заголовки запросов.
- Serverspace API base: `https://api.serverspace.ru` (auth header: `X-API-KEY: ...`).
- Целевой VPS: `85.198.90.128`
- Принцип исполнения: сначала `verify` (код/зависимости/тесты/линт на VPS), затем `run` (прогоны), затем `fetch` (забрать результаты обратно в игнорируемую директорию).

## Remote runs (Serverspace)
- API docs: https://docs.serverspace.ru/public_api.html
- Скрипт: `coint4/scripts/remote/run_server_job.sh`
- Переменные: `SERVSPACE_API_KEY`, `SERVER_ID` (или `SERVER_NAME`), `SERVER_IP` (по умолчанию `85.198.90.128`)
- Опции: `SKIP_POWER=1`, `STOP_AFTER=0/1`, `UPDATE_CODE=1/0`, `SYNC_UP=0/1`, `STOP_VIA_SSH=0/1`, `SYNC_BACK=1/0`, `SYNC_PATHS`, `SSH_KEY`, `SERVER_REPO_DIR`, `SERVER_WORK_DIR`, `LOCAL_REPO_DIR`
- Примечание: `STOP_AFTER=1` (default) = **автоматически выключить VPS** после выполнения команды. Не ставить `STOP_AFTER=0` без явной причины.
- Примечание: `SYNC_UP=1` = перед запуском **засинкать на VPS все tracked файлы** из локального репо (когда локальный `main` не запушен в origin, или `git pull` на VPS не принесёт нужные коммиты).
- Примечание: если нет `SERVSPACE_API_KEY`, можно запускать с `SKIP_POWER=1 STOP_VIA_SSH=1` (shutdown по SSH в конце, после sync_back).
- Пример (из `coint4/`):
  - `export SERVSPACE_API_KEY="***"; export SERVER_ID="***"; export SERVER_IP="85.198.90.128"`
  - `bash scripts/remote/run_server_job.sh bash scripts/optimization/watch_wfa_queue.sh --queue artifacts/wfa/aggregate/<group>/run_queue.csv`

## FAQ / грабли (частые вопросы по ходу прогонов)
- “Почему кажется, что прогоны запускаются локально?”
  - Локально мы обычно делаем только “подготовку”: генерируем конфиги/очереди, пересобираем rollup, ранжируем.
  - Тяжёлое исполнение WFA/оптимизаций должно идти **только** через `coint4/scripts/remote/run_server_job.sh` на `85.198.90.128`.
  - Быстрый способ снять сомнения: добавить в команду `hostname`/`uname -a`, чтобы это попало в stdout:
    - `bash scripts/remote/run_server_job.sh bash -lc 'echo RUN_HOST=$(hostname); bash scripts/optimization/watch_wfa_queue.sh --queue artifacts/wfa/aggregate/<group>/run_queue.csv'`
- “Как сделать, чтобы проект на VPS точно совпадал с тем, что здесь?”
  - Использовать `SYNC_UP=1` (он rsync’ит **только tracked** файлы, то есть `git ls-files`).
  - Новые файлы (очереди, конфиги, новые скрипты) нужно минимум `git add ...` (лучше сразу коммит) перед запуском с `SYNC_UP=1`, иначе они не попадут на VPS.
  - Рекомендация: перед VPS-прогоном держать чистый `git status` (sync_back может перезаписать файлы внутри `docs/`, `coint4/artifacts/`, `coint4/outputs/`).
- “Ранкер/rollup не видит новые результаты или показывает пусто”
  - После sync_back обязательно пересобрать индекс:
    - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`
  - Если прогоны шли не через watcher/queue-runner, статусы в `run_queue.csv` могут остаться `planned`:
    - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/<group>/run_queue.csv`
- “Сгенерировал очередь с OOS окнами и sweep-параметрами, а получилось 3×3 окон (мусорные сочетания start/end)”
  - В `scripts/optimization/generate_configs.py` нужно zip’ать `walk_forward.start_date` + `walk_forward.end_date`, иначе будет декартово произведение.
  - Использовать: `--zip-keys walk_forward.start_date,walk_forward.end_date`
- “Как убедиться, что VPS реально выключился (и не жрёт деньги)?”
  - По умолчанию `STOP_AFTER=1` должен выключать VPS в конце remote job.
  - Важно: `run_server_job.sh` пытается выключить VPS даже если команда/SSH внутри job упали. Если нужно оставить VPS для дебага, явно ставить `STOP_AFTER=0`.
  - Проверка: после завершения команды `ssh root@85.198.90.128 "echo ok"` должен перестать отвечать.
  - Если выключение не сработало: немедленно выключить в панели/через API (и потом разобраться почему).
- “Почему в live используется `PaperTradingEngine`?”
  - Это историческое название класса. Факт реальных ордеров определяется `BYBIT_ENV`.
  - В коде есть явное предупреждение: `BYBIT_ENV=live` размещает реальные ордера на Bybit (`coint4/scripts/run_live.py`, `coint4/src/coint2/live/runner.py`).
- “Мы делаем paper trading?”
  - Нет. Не делаем и не планируем. Cutover: сразу в `live`.
- “Я случайно засветил ключ/токен”
  - Не вставлять ключи в чат/issue/логи. Если ключ где-либо засветили, считать скомпрометированным и перевыпустить.

## Live trading / cutover
- Paper trading (demo/testnet) **не делаем и не планируем**.
- Прод-конфиг: `configs/prod_final_budget1000.yaml`
- План cutover: запуск **сразу в live** (без paper-этапа). Команда (из `coint4/`): `BYBIT_ENV=live BYBIT_API_KEY="***" BYBIT_API_SECRET="***" PYTHONPATH=src ./.venv/bin/python scripts/run_live.py --config configs/prod_final_budget1000.yaml --env live`
- Мониторинг: логи в `coint4/artifacts/live/logs`, состояние `coint4/artifacts/live/state.json`, снапшоты через `coint4/scripts/extract_live_snapshot.py` (обновляет `coint4/artifacts/live/LIVE_DASHBOARD.md`).

## Автономный режим (по умолчанию)
- Если нет явного стоп-сигнала, агент **сам выбирает следующий шаг** и выполняет работу до достижения цели.
- Агент **не ждёт, что человек подскажет очередь/команду**: сам находит актуальные `run_queue.csv`, поднимает VPS, запускает heavy-run, синкает результаты, анализирует и генерирует следующий батч.
- Уточняющие вопросы задаются только когда без них невозможно продолжать (нет доступа/ключей/инфра блокер/неоднозначный выбор с высоким риском).
- Формат отчётности: короткие апдейты по факту (что запущено/что завершилось/ключевые метрики/что дальше).

## Язык
- Всегда общаемся на русском языке.

## Правило непрерывного цикла (после замечаний о ручных запусках)
- После каждого срабатывания/неуспеха очереди делать только анализ полного контекста: последние логи `queue.log`, `candidate`/`daemon`-логи, статус VPS, и только после этого менять тактику.
- Ручные команды запуска `heavy-run`/`run_wfa_queue*` **не использовать по умолчанию**; запускать только через непрерывный оркестратор (автоматический селектор очереди + авто-триаж состояния).
- Перед любым перезапуском очереди:
  - сверить последнюю причину статуса из `strategy_metrics/equity_curve/errors`;
  - при системной проблеме (например, зависшие `running` без процесса) зафиксировать меру устранения и применить её в оркестраторе/конфиге;
  - только после этого стартовать следующий этап автоматически.
- Если пользователь просит ускорить/замкнуть цикл, сохранять правило «анализ → изменение параметров/модели → автоматический перезапуск». 

- 2026-03-04: автономный драйвер WFA теперь использует `urgency_score` (stalled*100 + running*20 + age_minutes*0.1), периодический `sync_queue_status` при наблюдаемом снижении pending, `stale-running watchdog` по mtime `results_dir` (STALE_RUNNING_SEC=900, минимум 60), и адаптивный backoff в idle (30→120→300).
- Запуск очередей по-прежнему идёт только через `autonomous_wfa_driver.sh`/`run_wfa_queue_powered` с `--wait-completion false`, без ручного запуска очередей по умолчанию.

- 2026-03-04: автономный драйвер WFA обновлён: добавлен `busy`-throttle (повторный skip одинаковой очереди без прогресса -> throttled sync/retry policy), динамический `--parallel` по размеру очереди/ETA/нагрузке, и авто-классификация root-cause (`NETWORK/DATA/MODEL/TIMEOUT/UNKNOWN`) перед стартом очереди с адаптивным `max_retries`.
- 2026-03-04: в `stale_running` убран внешний путь на VPS (inline fallback), чтобы watchdog зависших `running` работал и при отсутствии `/opt/coint4/coint4/scripts/optimization/_autonomous_stale_running.py`.
