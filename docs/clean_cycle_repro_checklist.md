# Clean-цикл (SYNC_UP=1 -> VPS run -> sync_back): чек-лист воспроизводимости

Цель: повторяемый “clean” цикл исполнения WFA/очереди на VPS `85.198.90.128` с синхронизацией только нужных файлов, без случайного попадания тяжёлых артефактов в Git.

## 0) Локальный preflight (перед VPS)

1. Убедиться, что вы в корне репо: `pwd` -> `/home/claudeuser/coint4`.
2. Проверить, что не стаджены тяжёлые/генерируемые артефакты:
   - `bash coint4/scripts/remote/verify_clean_cycle.sh`
3. Если планируется `SYNC_UP=1`, убедиться, что под ключевыми путями нет untracked (они не попадут на VPS при `SYNC_UP=1`):
   - `bash coint4/scripts/remote/verify_clean_cycle.sh --sync-up`
4. Если добавляли новые очереди/конфиги/доки: они должны быть tracked (иначе `SYNC_UP=1` их не синкнет).
   - Проверка: `git status --porcelain`

## 1) Проверки кода (локально)

Из корня репозитория:

- `make ci`

Важно: на этом сервере `146.103.41.248` тяжёлые прогоны не запускаем.

## 2) Remote run на VPS (рекомендуемый сценарий)

Запускать из `coint4/` (app-root).

1. Команда (пример под очередь):
   - `SYNC_UP=1 STOP_AFTER=1 bash scripts/remote/run_server_job.sh bash -lc 'echo RUN_HOST=$(hostname); bash scripts/optimization/watch_wfa_queue.sh --queue artifacts/wfa/aggregate/<group>/run_queue.csv'`

Примечания:

- `SYNC_UP=1` синкает на VPS только tracked файлы (`git ls-files` + `rsync`).
- `STOP_AFTER=1` (default) должен выключать VPS по завершении.
- `SYNC_BACK=1` (default) забирает назад `docs` и `coint4/artifacts` (и др. пути из `SYNC_PATHS`).

## 3) После sync_back (локально)

1. Если прогоны шли не через queue-runner/watcher и в `run_queue.csv` остались `planned`:
   - `cd coint4`
   - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/<group>/run_queue.csv`
2. Пересобрать rollup индекс:
   - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`

## 4) Git hygiene (перед коммитом)

1. Никогда не делать `git add -A`. Добавлять только конкретные файлы.
2. Перед коммитом:
   - `git status --porcelain`
   - `git diff --cached --name-only`
   - `bash coint4/scripts/remote/verify_clean_cycle.sh`

## 5) Быстрая проверка, что VPS выключился

После завершения remote job:

- `ssh root@85.198.90.128 "echo ok"` должно перестать отвечать (если `STOP_AFTER=1` реально отработал).

