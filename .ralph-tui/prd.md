# PRD: VPS loop (verify -> run -> fetch) + единый Sharpe для coint4

## Objective
Полностью автономный цикл через `ralph-tui` + `codex`, который:
- приводит Sharpe к одной канонической методике (код + тесты + утилиты),
- готовит инфраструктуру для прогона исторических WFA-конфигов на VPS `85.198.90.128` (без лишних окон; tmux/single-session),
- сначала проверяет корректность кода на VPS (репо/зависимости/линт/тесты),
- запускает прогоны и затем забирает результаты обратно в **игнорируемую** директорию.

## Constraints
- Никогда не коммитить: `.secrets/**`, `coint4/artifacts/**`, `coint4/outputs/**`, `outputs/**`, `.ralph-tui/iterations/**`, `*.pid`, `*.log`.
- Никаких `git add -A`. Стадить только явные файлы.
- 1 таск = 1 маленький атомарный коммит. Сообщения только: `chore(ralph): ...`, `feat: ...`, `fix: ...`.
- Не запускать тяжёлые WFA/бэктесты на этом хосте. Тяжёлое только на VPS.

---

### US-001: Короткий context pack (как работать и где что лежит)
**Priority:** P1

- [ ] Добавлен `docs/context_pack.md` (коротко, без простыней):
  - каноническая формула Sharpe (annualization, ddof, rf),
  - команды quick-check (локально/на VPS): `make ci`,
  - pipeline: `verify -> run -> fetch`,
  - куда складываются fetched результаты (игнорируемо),
  - правила: не тащить секреты/артефакты в git.

### US-002: Канонический модуль Sharpe
**Priority:** P1

- [ ] Добавлен модуль `coint4/src/coint2/core/sharpe.py`.
- [ ] В модуле явно зафиксированы:
  - Sharpe = sqrt(periods_per_year) * mean(excess_returns) / std(excess_returns, ddof=1)
  - risk_free_rate по умолчанию 0 (в единицах per-period)
  - поведение при std=0: возвращаем 0.0 (не NaN/inf)
- [ ] Есть функция для Series returns и для equity-series (pct_change).

### US-003: Performance API использует канонический Sharpe
**Priority:** P1
**Depends on:** US-002

- [ ] `coint4/src/coint2/core/performance.py` использует `coint2.core.sharpe` (единый ddof/annualization/edge-cases).
- [ ] Убрана внутренняя несогласованность ddof между функциями Sharpe.

### US-004: Тесты на Sharpe (минимум 3 кейса)
**Priority:** P1
**Depends on:** US-002

- [ ] Добавлен `coint4/tests/unit/test_sharpe.py`.
- [ ] Минимум 3 кейса:
  - константные returns -> Sharpe = 0.0
  - известный массив returns -> ожидаемое значение (с допуском)
  - проверка, что ddof=1 и annualization применяются как в спецификации

### US-005: Run index пересчитывает Sharpe через канонический модуль
**Priority:** P2
**Depends on:** US-002

- [ ] `coint4/src/coint2/ops/run_index.py` использует `coint2.core.sharpe` для пересчёта Sharpe из `equity_curve.csv`.
- [ ] Поведение сохраняет текущую канонизацию: `sharpe_ratio_abs` (computed) предпочтительнее raw, если computed доступен.

### US-006: Утилита пересчёта Sharpe по артефактам (локально и на VPS)
**Priority:** P2
**Depends on:** US-002

- [ ] Добавлен `tools/recompute_sharpe.py`.
- [ ] Скрипт умеет:
  - принимать `--runs-glob` или `--runs-root`,
  - читать `equity_curve.csv`,
  - считать Sharpe канонически,
  - писать summary CSV/JSON в указанный output (по умолчанию в игнорируемый `coint4/outputs/`).
- [ ] По умолчанию скрипт НЕ переписывает артефакты; только отчёт.

### US-007: Secret scan перед коммитом (без хранения ключа в коде)
**Priority:** P1

- [ ] Добавлен `tools/dev/secret_scan_staged.sh`.
- [ ] Скрипт проверяет staged изменения и падает (exit 1), если:
  - в staged попали пути `.secrets/**` или другие запрещённые артефактные директории,
  - в staged diff встречается `X-API-KEY`, `SERVSPACE_API_KEY=` или 64-hex строка (консервативная защита).
- [ ] Скрипт НЕ печатает секреты (только нейтральные сообщения).

### US-008: Serverspace API client (минимальный)
**Priority:** P1

- [ ] Добавлен `tools/serverspace_api.py`.
- [ ] API base по умолчанию: `https://api.serverspace.ru/api/v1`.
- [ ] Ключ берётся из `SERVSPACE_API_KEY` или из `.secrets/serverspace_api_key` (если env отсутствует).
- [ ] Команды/режимы:
  - list servers
  - find server by IP `85.198.90.128` (печатает id + name + state)
  - power on / shutdown (POST)
- [ ] В логах/выводе НЕТ значения ключа.

### US-009: SSH exec helper (tmux/single-session)
**Priority:** P1

- [ ] Добавлен `tools/vps_exec.py`.
- [ ] Поддержка:
  - запуск команды по SSH (`bash -lc`),
  - опционально: запуск команды внутри tmux session (создать, если нет),
  - режим проверки существования tmux session.
- [ ] По умолчанию: user=root, host=85.198.90.128, key=`~/.ssh/id_ed25519`.

### US-010: Sync code to VPS (без артефактов)
**Priority:** P1
**Depends on:** US-009

- [ ] Добавлен `tools/sync_to_vps.py`.
- [ ] Синхронизирует на VPS только нужные директории (код/скрипты/тесты/конфиги), исключая артефакты и `.secrets`.
- [ ] На VPS пишет `SYNCED_FROM_COMMIT.txt` с локальным git SHA.
- [ ] Скрипт возвращает non-zero при ошибке rsync/ssh.

### US-011: VPS verify stage (быстрые проверки корректности)
**Priority:** P1
**Depends on:** US-009, US-010

- [ ] Добавлен `tools/vps_verify.py`.
- [ ] Запускает на VPS быстрые проверки проекта (без WFA):
  - `make ci` (или эквивалент, если make недоступен),
  - логирует только нейтральную информацию.
- [ ] Скрипт фейлится с понятной ошибкой, если зависимости не установлены/команда не найдена.

### US-012: Manifest исторических прогонов (из run_queue.csv)
**Priority:** P2

- [ ] Добавлен `tools/build_wfa_manifest.py`.
- [ ] По умолчанию:
  - находит `coint4/artifacts/wfa/aggregate/**/run_queue.csv` локально,
  - выбирает статусы `planned,stalled` (без перезапуска completed),
  - пишет JSON manifest в игнорируемый `coint4/outputs/`.
- [ ] Есть флаг `--include-completed` для полного прогона при необходимости.

### US-013: VPS run stage (один tmux session, без размножения окон)
**Priority:** P1
**Depends on:** US-009, US-012

- [ ] Добавлен `tools/vps_run_wfa.py`.
- [ ] Скрипт:
  - создаёт (или переиспользует) один tmux session (например `coint4-wfa`),
  - запускает последовательный проход по queue-файлам из manifest (без множества tmux окон),
  - пишет sentinel-файл `WFA_RUN_DONE.txt` в `coint4/outputs/` на VPS по завершению.

### US-014: VPS fetch stage (без перетаскивания гигабайт по умолчанию)
**Priority:** P1
**Depends on:** US-009

- [ ] Добавлен `tools/vps_fetch_results.py`.
- [ ] По умолчанию скачивает только лёгкие результаты:
  - обновлённые `run_queue.csv`/логи watcher,
  - rollup `run_index.*` и sharpe summary CSV/JSON,
  - любые отчёты из `coint4/outputs/`.
- [ ] Скачивание кладёт в `coint4/outputs/vps_fetch/<timestamp>/` (игнорируемо).
- [ ] Есть флаг `--include-runs` для полного rsync `artifacts/wfa/runs/**` (off by default).

### US-015: End-to-end pipeline (power -> sync -> verify -> run -> fetch)
**Priority:** P1
**Depends on:** US-008, US-010, US-011, US-013, US-014

- [ ] Добавлен `tools/vps_pipeline.py`.
- [ ] Команда вида `coint4/.venv/bin/python tools/vps_pipeline.py --all` делает:
  - power-on (Serverspace API), wait-for-ssh,
  - sync code,
  - verify stage,
  - start run stage в tmux,
  - fetch (если sentinel найден; иначе печатает, как проверить статус и повторить fetch).
- [ ] Никаких секретов в выводе.
