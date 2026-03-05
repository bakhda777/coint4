# OpenClaw Guardrails

Цель: уменьшить частые инфраструктурные и оркестрационные ошибки при автономных прогонах.

## MUST
- Модели: использовать только `openai-codex/gpt-5.3-codex-spark` или `openai-codex/gpt-5.3-codex`.
- Запрещено: `openai-codex/gpt-5.3-spark` (invalid/unknown model).
- Python: запускать только через `PYTHONPATH=src ./.venv/bin/python ...` из `coint4/`.
- Перед запуском проверять пути/файлы (`run_queue.csv`, конфиги, скрипты) и права доступа.
- Не писать в завершённый процесс: при `completed/exitCode!=null` создавать новый процесс.
- Валидировать аргументы команд: не допускать `null bytes` и битых путей.
- После ручных WFA запусков обязательно:
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/<group>/run_queue.csv`
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`

## Preflight Checklist
- Находимся в `coint4/` и есть `.venv`:
  - `pwd`
  - `test -x ./.venv/bin/python`
- Проверяем модель в конфиге OpenClaw: whitelist из раздела MUST.
- Проверяем входные файлы:
  - `test -f artifacts/wfa/aggregate/<group>/run_queue.csv`
  - `test -f scripts/optimization/watch_wfa_queue.sh`
- Проверяем права на запись в рабочие/лог-директории.

## Частые ошибки и фиксы
- `Unknown model: openai-codex/gpt-5.3-spark`
  - Причина: неверный model id.
  - Фикс: переключить на `openai-codex/gpt-5.3-codex-spark` (или `.../gpt-5.3-codex`).
- `python: command not found`
  - Причина: вызов системного `python` без venv.
  - Фикс: всегда `PYTHONPATH=src ./.venv/bin/python ...`.
- `write after end`
  - Причина: запись в уже завершённый процесс.
  - Фикс: проверять статус процесса; для новых команд создавать новый процесс.
- `... null bytes`
  - Причина: повреждённые аргументы/командная строка.
  - Фикс: санитизировать параметры и пересобрать команду.
- `EACCES`
  - Причина: недостаточные права на конфиг/логи/директории.
  - Фикс: исправить owner/permissions и повторить запуск.
- `ENOENT` / `File not found`
  - Причина: неверный путь, файл не создан/не синкнут.
  - Фикс: preflight проверка `test -f`, при remote запуске использовать `SYNC_UP=1`.
