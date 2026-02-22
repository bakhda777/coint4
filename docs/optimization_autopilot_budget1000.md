# Budget $1000: автопилот оптимизации (VPS WFA -> постпроцесс -> следующий шаг)

Цель: убрать ручной цикл “запустил -> жду -> не вижу что закончилось -> прошу Codex проанализировать -> генерю следующую очередь -> снова запускаю”.

Автопилот делает маленькие микросвипы (coordinate ascent) вокруг текущего лучшего конфига:
- генерирует очереди WFA (multi-window, holdout+stress),
- гоняет их на VPS `85.198.90.128` через `coint4/scripts/remote/run_server_job.sh` (по умолчанию `STOP_AFTER=1`),
- после sync_back пересчитывает `canonical_metrics.json`, пересобирает rollup `run_index.csv`,
- выбирает лучший вариант по robust-метрике с учётом DD (гейт или penalty) и минимального P&L по окнам (по умолчанию не ниже 0),
- останавливается, когда полный круг по knobs не даёт улучшений.

## Предусловия

Из корня репозитория:
1) Зависимости:
   - `make setup`
2) Доступ к VPS:
   - SSH ключ (по умолчанию `~/.ssh/id_ed25519`) должен подходить для `root@85.198.90.128`
3) Serverspace API key (для power on/off):
   - `SERVSPACE_API_KEY` в env, или файл `.secrets/serverspace_api_key` (gitignored, `chmod 600`)

Важно:
- На этом сервере тяжёлые прогоны не запускаем. Автопилот запускает heavy только на VPS.
- `SYNC_UP=1` загружает на VPS только tracked файлы (`git ls-files`). Автопилот по умолчанию делает `git add` для сгенерённых очередей/конфигов/состояния (но не для `runs_clean/**`).

## Конфиг автопилота

Файл: `coint4/configs/autopilot/budget1000.yaml`

Что обычно правят:
- `base_config`: с какого конфига стартуем (можно указать уже “лучший на сейчас”).
- `windows`: OOS окна (paired start/end).
- `selection`: DD-гейт (`max_dd_pct`) или soft penalty (`dd_target_pct` + `dd_penalty`), плюс минимум P&L (`min_pnl`).
- `search.knobs`: какие параметры крутить и шаги.
- `run_group_prefix`: как будут называться очереди/результаты.

## Запуск

1) Dry-run (только сгенерировать следующую очередь и выйти):
   - `cd coint4`
   - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/autopilot_budget1000.py --config configs/autopilot/budget1000.yaml --dry-run --reset`

2) Полный запуск (запускает VPS, ждёт завершения очередей, постпроцессит, продолжает до stop-condition):
   - `cd coint4`
   - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/autopilot_budget1000.py --config configs/autopilot/budget1000.yaml --reset`

3) Resume (если прервалось на середине):
   - `cd coint4`
   - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/autopilot_budget1000.py --config configs/autopilot/budget1000.yaml --resume`

## Где смотреть результат

Очереди и маленькие артефакты (коммитабельно):
- `coint4/artifacts/wfa/aggregate/<run_group>/run_queue.csv`
- `coint4/configs/budget1000_autopilot/<run_group>/*.yaml`
- контроллер: `coint4/artifacts/wfa/aggregate/<controller_group>/state.json`

Тяжёлые результаты (не коммитить):
- `coint4/artifacts/wfa/runs_clean/<run_group>/**`

Итоговый отчёт:
- `docs/budget1000_autopilot_final_YYYYMMDD.md`

## (Опционально) Запуск через ralph-tui

Если используешь `ralph tui` как трекер/оркестратор Codex-задач, можно запускать автопилот как 1-2 user stories:
1) Подложить PRD в canonical path, который читает ralph:
   - `cp tasks/budget1000_autopilot/prd_budget1000_autopilot.ralph.json .ralph-tui/prd.json`
2) Запустить `ralph tui` (дальше он будет гонять Codex agent по задачам из `.ralph-tui/prd.json`).

Примечание: автопилот сам обновляет `state.json` и пишет финальный отчёт; ralph нужен только как UI/трекер.

## Ручной постпроцесс (если нужно отдельно)

Если очередь запускалась вручную, можно локально привести всё в порядок:
- `cd coint4`
- `PYTHONPATH=src ./.venv/bin/python scripts/optimization/postprocess_queue.py --queue artifacts/wfa/aggregate/<run_group>/run_queue.csv --bar-minutes 15 --overwrite-canonical --build-rollup`
