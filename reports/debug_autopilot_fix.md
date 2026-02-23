# Debug: autopilot_loop / ralph-tui зацикливание (2026-02-23)

## Симптом
- Codex-воркер “тонет” в старых ретро/логах/прогрессе и пишет длинные простыни, из‑за чего Ralph долго не видит завершение задачи.

## Текущее состояние (до фикса)
- `tools/autopilot_loop.sh` запускает Ralph так:
  - `ralph-tui run --headless --no-setup --no-sandbox --serial --tracker beads --epic "${EPIC_ID}"`
  - Проблема: `--no-sandbox` конфликтует с идеей “Codex full-auto + sandbox=workspace-write” в конфиге (и просто лишний).

- `.ralph-tui/config.toml` (релевантное):
  - `[agentOptions] fullAuto=true`
  - `[agentOptions] sandbox="workspace-write"`
  - `prompt_template` не задан (значит работает шаблон трекера по умолчанию).

## Что реально поддерживает ralph-tui (по dist/cli.js)
- Нативных ключей для игнора/исключения путей из “контекста” не найдено (поиск по `contextIgnore|excludeGlobs|ignorePaths|excludePaths` пустой).
- Есть поддержка `prompt_template` (кастомный handlebars-шаблон промпта).
- Completion marker, который использует движок:
  - `PROMISE_COMPLETE_PATTERN = /<promise>\\s*COMPLETE\\s*<\\/promise>/i`
  - Т.е. Ralph считает итерацию “done”, когда stdout агента содержит строку `<promise>COMPLETE</promise>` (регистр не важен).

## Наблюдение про parallel
- В `StoredConfigSchema` есть секция `parallel`, но в `mergeConfigs(...)` она не мерджится (в результате `ralph-tui config show` её не показывает).
- Поэтому для гарантированно последовательного режима нужен флаг CLI `--serial` (а не только конфиг).

