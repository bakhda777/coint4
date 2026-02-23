# OPERATING MODE: CODE-ONLY / NO-EXEC

Этот репозиторий работает в режиме `CODE-ONLY / NO-EXEC` по умолчанию.

## Что разрешено

- Изменять код, конфиги, очереди и документацию.
- Запускать лёгкие проверки (`dry-run`, `make lint`, `make test`, unit/smoke проверки).
- Готовить run_queue/config-пакеты для удалённого исполнения.

## Что запрещено по умолчанию

- Запускать heavy compute локально (WFA/fullcpu/Optuna trial loops/долгие batch).
- Автоматически стартовать heavy через фоновые службы без явного ручного разрешения.
- Запускать автозапуск из `ralph`/агентов без операторского подтверждения окружения.

## Heavy-run policy (v1)

Heavy запуск допустим только при одновременном выполнении всех условий:

1. `ALLOW_HEAVY_RUN=1`.
2. Хост входит в allowlist (`HEAVY_HOSTNAME_ALLOWLIST`, по умолчанию: `85.198.90.128,coint`).
3. Ресурсы не ниже порога (`HEAVY_MIN_RAM_GB`, `HEAVY_MIN_CPU`; по умолчанию: `28 GiB`, `8 CPU`).
4. Запуск выполняется вручную через канонический runner:
   - `coint4/scripts/batch/run_heavy_queue.sh`.

При нарушении любого условия действует fail-fast с non-zero exit.

## Обязательный постпроцесс после heavy блока

Postprocess запускается вручную отдельным шагом:

1. `PYTHONPATH=src ./.venv/bin/python scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/<run_group>/run_queue.csv`
2. `PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup --no-auto-sync-status`
3. `PYTHONPATH=src ./.venv/bin/python scripts/optimization/rank_multiwindow_robust_runs.py --run-index artifacts/wfa/aggregate/rollup/run_index.csv --contains <run_group_or_tag> --fullspan-policy-v1 --min-windows 1 --min-trades 200 --min-pairs 20 --max-dd-pct 0.50 --min-pnl 0 --initial-capital 1000 --tail-quantile 0.20 --tail-q-soft-loss-pct 0.03 --tail-worst-soft-loss-pct 0.10 --tail-q-penalty 2.0 --tail-worst-penalty 1.0 --tail-worst-gate-pct 0.20`

