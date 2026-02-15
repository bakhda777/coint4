# PRD: Автономная петля (Ralph Loop) для coint4

## Objective
Сделать воспроизводимый автономный цикл разработки через `ralph-tui` + `codex`:
- маленькие, осмысленные изменения,
- один таск = один коммит,
- короткий кросс-итерационный контекст в `.ralph-tui/progress.md`,
- никаких тяжёлых WFA/бэктестов на этой машине.

## Constraints
- Запрещены тяжёлые прогоны WFA/оптимизаций/долгие бэктесты.
- Не коммитить артефакты/выгрузки/выходы бэктестов.
- Нельзя использовать `git add -A`; стадить только релевантные файлы.
- Сообщения коммитов только: `chore(ralph): ...`, `feat: ...`, `fix: ...`.

---

### US-001: Документация Ralph Loop (как запускать и какие правила)
**Priority:** P1

- [ ] Добавлен документ `docs/ralph_loop.md` с описанием:
  - как запускать headless loop,
  - где лежат `.ralph-tui/prd.md`/`.ralph-tui/prd.json` и `.ralph-tui/progress.md`,
  - правила маленьких диффов и коммитов,
  - запрет на коммит артефактов.
- [ ] В документе есть точная команда запуска:
  - `ralph-tui run --no-tui --prd ./.ralph-tui/prd.json --agent codex --prompt ./.ralph-tui/templates/json.hbs`

### US-002: Make target для smoke-тестов
**Priority:** P1

- [ ] В корневом `Makefile` добавлена цель `make smoke`, которая запускает `pytest -q -m smoke` в `coint4/`.
- [ ] `make help` показывает новую цель.

### US-003: Make target для preflight
**Priority:** P2

- [ ] В корневом `Makefile` добавлена цель `make preflight`, которая запускает `coint4/scripts/run_preflight.py`.
- [ ] Цель использует `coint4/.venv/bin/python` (как и остальные цели Makefile).

### US-004: CLI-скрипт валидации прод-конфига
**Priority:** P1

- [ ] Добавлен скрипт `coint4/scripts/validate_config.py`.
- [ ] Команда `coint4/.venv/bin/python scripts/validate_config.py --config configs/main_2024.yaml` возвращает exit code 0 и печатает результат.
- [ ] Для несуществующего файла или неподдерживаемого расширения скрипт возвращает exit code 1.

### US-005: CLI-скрипт валидации search space
**Priority:** P2

- [ ] Добавлен скрипт `coint4/scripts/validate_search_space.py`.
- [ ] Команда `coint4/.venv/bin/python scripts/validate_search_space.py --path configs/search_spaces/fast.yaml` возвращает exit code 0 и печатает результат.
- [ ] Для несуществующего файла скрипт возвращает exit code 1.

### US-006: Smoke-тесты для новых validation CLI
**Priority:** P1
**Depends on:** US-004, US-005

- [ ] Добавлены smoke-тесты, которые проверяют exit code для `validate_config.py` и `validate_search_space.py`.
- [ ] Тесты не требуют внешних данных и укладываются в лимиты smoke.

### US-007: Make target validate-config
**Priority:** P2
**Depends on:** US-004

- [ ] В `Makefile` добавлена цель `make validate-config`.
- [ ] По умолчанию валидируется `coint4/configs/main_2024.yaml` (можно переопределить переменной `CONFIG=...`).

### US-008: Make target validate-search-space
**Priority:** P3
**Depends on:** US-005

- [ ] В `Makefile` добавлена цель `make validate-search-space`.
- [ ] По умолчанию валидируется `coint4/configs/search_spaces/fast.yaml` (можно переопределить переменной `PATH=...` или `SPACE=...`).

### US-009: Скрипт проверки git-гигиены (не тащим генерируемое в git)
**Priority:** P1

- [ ] Добавлен скрипт `coint4/scripts/dev/check_tracked_generated.sh`.
- [ ] Скрипт падает (exit 1), если `git ls-files` находит что-то в:
  - `coint4/artifacts/`, `coint4/outputs/`, `outputs/`, `.ralph-tui/iterations/`, `.ralph-tui/progress.md`, а также любые `*.pid`/`*.log`.
- [ ] В чистом репо скрипт возвращает exit 0.

### US-010: Make target hygiene
**Priority:** P2
**Depends on:** US-009

- [ ] В `Makefile` добавлена цель `make hygiene`, запускающая `coint4/scripts/dev/check_tracked_generated.sh`.

### US-011: Обновить testing guide с новыми командами
**Priority:** P3

- [ ] `docs/testing_guide.md` обновлён: добавлены `make smoke`, `make preflight`, `make validate-config`, `make hygiene`.

### US-012: Обновить quickstart с validate/smoke
**Priority:** P3

- [ ] `docs/quickstart.md` обновлён: добавлены быстрые команды `make smoke` и `make validate-config`.

### US-013: README: упомянуть Ralph loop docs и быстрые проверки
**Priority:** P3

- [ ] В `README.md` добавлена короткая ссылка на `docs/ralph_loop.md`.
- [ ] В quickstart/Dev commands упомянуты `make smoke` и `make hygiene`.

### US-014: CI guard: запуск hygiene в GitHub Actions
**Priority:** P4
**Depends on:** US-010

- [ ] В `.github/workflows/tests.yml` добавлен шаг `make hygiene` (достаточно в одном job).

### US-015: Smoke-тест для dev hygiene скрипта
**Priority:** P2
**Depends on:** US-009

- [ ] Добавлен smoke-тест, который запускает `coint4/scripts/dev/check_tracked_generated.sh` и ожидает exit code 0.
