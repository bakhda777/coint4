# Clean Cycle TOP-10: схема оптимизации (процесс, запреты, score и фильтры)

Этот документ фиксирует **канонический** clean-цикл TOP-10 (baseline + sweeps): где лежат результаты, какой инвариант по окнам, как строится clean rollup, и где действует запрет “смотреть на старую партию”.

Примечание про пути:
- В тексте ниже пути пишутся от корня репозитория с префиксом `coint4/...` (чтобы не путать с `docs/`).
- Команды ниже предполагают запуск из app-root `coint4/` (поэтому в командах префикса `coint4/` нет).

## Термины и источники истины

- **Старая партия (legacy)**:
  - результаты: `coint4/artifacts/wfa/runs/**`
  - глобальный rollup-индекс: `coint4/artifacts/wfa/aggregate/rollup/run_index.*`
- **Clean-контур**:
  - результаты baseline/sweeps: `coint4/artifacts/wfa/runs_clean/<cycle>/...`
  - агрегаты (манифесты/очереди/rollup): `coint4/artifacts/wfa/aggregate/clean_cycle_top10/<cycle>/...`
  - конфиги: `coint4/configs/clean_cycle_top10/**`
- **Источник истины по метрикам** (для ранжирования): `canonical_metrics.json`, пересчитанный из `equity_curve.csv`.
  - пересчёт: `coint4/scripts/optimization/recompute_canonical_metrics.py`
  - формула score: `coint4/scripts/optimization/clean_cycle_top10/scoring.py`

## Канонические пути (текущий cycle)

Источник истины по значениям: `coint4/scripts/optimization/clean_cycle_top10/definitions.py`.

- `CYCLE_NAME`: `20260215_clean_top10`
- `CLEAN_ROOT`: `coint4/artifacts/wfa/runs_clean/20260215_clean_top10`
- `BASELINE_DIR`: `coint4/artifacts/wfa/runs_clean/20260215_clean_top10/baseline_top10`
- `OPT_DIR`: `coint4/artifacts/wfa/runs_clean/20260215_clean_top10/opt_sweeps`
- `CLEAN_AGG_DIR`: `coint4/artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10`

Типовые файлы в `CLEAN_AGG_DIR` (маленькие, для воспроизводимости):
- `run_index_snapshot.json` (sha256 + shape для `run_index.csv`)
- `baseline_manifest.json` (initial TOP-10, сделан из старой партии; использовать только как seed)
- `baseline_run_queue.csv` (10 baseline runs в clean-контуре; это baseline input для clean rollup)
- `sweeps_manifest.json` (описание sweep-конфигов/параметров; без results_dir)
- `sweeps_run_queue.csv` (sweeps runs в clean-контуре; это sweep input для clean rollup)
- `rollup_clean_cycle_top10.(csv|md)` (clean rollup на canonical_*; строится по queue-файлам, не по старой партии)

## FIXED_WINDOWS.walk_forward (инвариант)

Все baseline и sweeps внутри clean-цикла должны иметь **одинаковый** `walk_forward`. Это инвариант clean-цикла.

Источник истины: `coint4/scripts/optimization/clean_cycle_top10/definitions.py`.

```json
{
  "start_date": "2024-05-01",
  "end_date": "2024-12-31",
  "training_period_days": 90,
  "testing_period_days": 30,
  "step_size_days": 30,
  "max_steps": 5,
  "gap_minutes": 15,
  "refit_frequency": "weekly"
}
```

Guardrails:
- Sweep по `walk_forward.*` запрещён (иначе это уже другой эксперимент, не сравнимый внутри clean-цикла).
- Для queue-прогонов `walk_forward.max_steps` должен быть **явно задан** и `<= 5` (это проверяет watcher/queue tooling).

## Запрет: “смотреть на старую партию”

Старая партия используется **только** в самом начале цикла, чтобы выбрать initial TOP-10 и зафиксировать вход:
- snapshot `run_index.csv` (sha256/shape) -> `run_index_snapshot.json`;
- TOP-N selection -> `baseline_manifest.json`.

Важно: `baseline_manifest.json` содержит `results_dir` из старой партии. Для clean rollup используем `baseline_run_queue.csv` и `sweeps_run_queue.csv` (они указывают на `runs_clean`).

После этого запрещено использовать старую партию для принятия решений внутри цикла:
- не ранжировать/не выбирать победителей по `coint4/artifacts/wfa/aggregate/rollup/run_index.*`;
- не “подсматривать” метрики/кривые/сводки из `coint4/artifacts/wfa/runs/**` для выбора следующего шага в clean-цикле.

После старта clean-цикла единственный источник для решений:
- `rollup_clean_cycle_top10.(csv|md)` в `CLEAN_AGG_DIR`, построенный только из `canonical_metrics.json` (baseline + sweeps внутри `runs_clean`).

Исключение (только дебаг, не для выбора победителя):
- `coint4/scripts/optimization/clean_cycle_top10/compare_metrics.py` можно использовать для sanity-check raw vs canonical, но итоговые решения всё равно принимаются по clean rollup на canonical_*.

## Процесс (A–F)

На этом сервере тяжёлые прогоны не запускать. Локально делаем только подготовку/проверки. Heavy execution запускается на VPS `85.198.90.128`.

### A) Зафиксировать старую партию и выбрать initial TOP-10 (подготовка)

Запускать из `coint4/`:

```bash
# 1) snapshot run_index.csv (freeze input старой партии)
PYTHONPATH=src ./.venv/bin/python scripts/optimization/clean_cycle_top10/snapshot_run_index.py \
  --run-index artifacts/wfa/aggregate/rollup/run_index.csv \
  --output artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/run_index_snapshot.json \
  --refuse-overwrite

# 2) select TOP-10 -> baseline_manifest.json
PYTHONPATH=src ./.venv/bin/python scripts/optimization/clean_cycle_top10/select_top10.py \
  --run-index artifacts/wfa/aggregate/rollup/run_index.csv \
  --output artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/baseline_manifest.json \
  --top-n 10 \
  --refuse-overwrite

# 3) validate manifest schema/paths/hashes + FIXED_WINDOWS compatibility
PYTHONPATH=src ./.venv/bin/python scripts/optimization/clean_cycle_top10/validate_manifest.py \
  --manifest artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/baseline_manifest.json \
  --use-definitions
```

### B) Инициализировать clean структуру (подготовка)

Создаёт директории и README (без тяжёлых вычислений):

```bash
PYTHONPATH=src ./.venv/bin/python scripts/optimization/clean_cycle_top10/init_clean_cycle.py --refuse-overwrite
```

### C) Сгенерировать baseline конфиги и очередь (подготовка)

Baseline YAML-ы генерируются строго из `baseline_manifest.json`, с принудительным `FIXED_WINDOWS.walk_forward`.

```bash
# dry-run: посмотреть, какие файлы будут записаны
PYTHONPATH=src ./.venv/bin/python scripts/optimization/clean_cycle_top10/make_baseline_configs.py \
  --manifest artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/baseline_manifest.json \
  --out-dir configs/clean_cycle_top10/baseline \
  --dry-run

# генерация baseline_run_queue.csv (10 запусков) из baseline-конфигов
# (используем общий queue builder, results_dir направляем в BASELINE_DIR)
PYTHONPATH=src ./.venv/bin/python scripts/optimization/clean_cycle_top10/build_sweeps_queue.py \
  --configs-dir configs/clean_cycle_top10/baseline \
  --opt-dir artifacts/wfa/runs_clean/20260215_clean_top10/baseline_top10 \
  --output artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/baseline_run_queue.csv \
  --dry-run
```

### D) Freeze baseline (после baseline batch)

После выполнения baseline прогонов (на VPS) baseline нужно “заморозить”, чтобы:
- нельзя было случайно перезаписать результаты baseline (sentinel + guard);
- можно было детерминированно воспроизвести цикл.

Команды (после того как baseline результаты и `canonical_metrics.json` готовы):

```bash
# записать sentinel BASELINE_FROZEN.txt
PYTHONPATH=src ./.venv/bin/python scripts/optimization/clean_cycle_top10/freeze_baseline.py \
  --manifest artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/baseline_manifest.json \
  --baseline-dir artifacts/wfa/runs_clean/20260215_clean_top10/baseline_top10 \
  --refuse-overwrite

# проверить целостность sentinel + FIXED_WINDOWS
PYTHONPATH=src ./.venv/bin/python scripts/optimization/clean_cycle_top10/verify_baseline_frozen.py \
  --baseline-dir artifacts/wfa/runs_clean/20260215_clean_top10/baseline_top10
```

### E) Sweeps от baseline (подготовка)

Sweep-конфиги генерируются **только** от одного baseline-конфига (обычно победителя baseline-ранжирования).

```bash
# dry-run: посмотреть сетку и что будет сгенерировано
PYTHONPATH=src ./.venv/bin/python scripts/optimization/clean_cycle_top10/make_sweep_configs.py \
  --baseline-config configs/clean_cycle_top10/baseline/b01_*.yaml \
  --sweep 'backtest.min_spread_move_sigma=[0.10,0.15,0.20]' \
  --out-dir configs/clean_cycle_top10/sweeps/min_spread_move_sigma \
  --manifest-out artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/sweeps_manifest.json \
  --dry-run

# очередь на sweeps
PYTHONPATH=src ./.venv/bin/python scripts/optimization/clean_cycle_top10/build_sweeps_queue.py \
  --sweeps-manifest artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/sweeps_manifest.json \
  --opt-dir artifacts/wfa/runs_clean/20260215_clean_top10/opt_sweeps \
  --output artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/sweeps_run_queue.csv \
  --dry-run
```

### F) Canonical metrics + clean rollup (после baseline/sweeps batch)

После выполнения прогонов (на VPS) пересчитываем canonical метрики и строим clean rollup.

Правила:
- Для каждого `results_dir` из `baseline_run_queue.csv` и `sweeps_run_queue.csv` должен быть `canonical_metrics.json` (пересчитан из `equity_curve.csv`).
- Clean rollup строим по **queue-файлам** (`baseline_run_queue.csv` + `sweeps_run_queue.csv`): они содержат `results_dir` и `status`.
  - `baseline_manifest.json` и `sweeps_manifest.json` в rollup напрямую не подаём: первое указывает на старую партию, второе не содержит `results_dir`.

## Score

Скалярная метрика (default):

`score = canonical_sharpe - lambda_dd * abs(canonical_max_drawdown_abs)`

- `canonical_max_drawdown_abs` измеряется в единицах equity (например, USD для budget1000) и обычно `<= 0`, поэтому берём `abs(...)`.
- `lambda_dd` параметризуется через CLI (default: `0.02`).

Источник истины по формуле: `coint4/scripts/optimization/clean_cycle_top10/scoring.py`.

## Сортировка

`coint4/scripts/optimization/clean_cycle_top10/build_clean_rollup.py` поддерживает 2 режима:

- `--sort-mode score` (default):
  - `score` desc
  - `canonical_sharpe` desc
  - `abs(canonical_max_drawdown_abs)` asc
  - `canonical_pnl_abs` desc
  - далее детерминированные tie-breakers (phase/run_name/ids/paths)
- `--sort-mode multi`:
  - `canonical_sharpe` desc
  - `abs(canonical_max_drawdown_abs)` asc
  - `canonical_pnl_abs` desc
  - далее детерминированные tie-breakers (phase/run_name/ids/paths)

## Фильтры (default)

Rollup строится только по:

- `status == completed`
- `canonical_metrics.json` присутствует (`canonical_metrics_present == true`)

Флаги для ослабления фильтров (только для дебага):

- `--include-noncompleted`
- `--include-missing-canonical`

## Команда (пример)

Запускать из `coint4/` (после того как в run_dir есть `canonical_metrics.json`):

```bash
PYTHONPATH=src ./.venv/bin/python scripts/optimization/clean_cycle_top10/build_clean_rollup.py \
  --baseline-manifest artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/baseline_run_queue.csv \
  --sweeps-manifest artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/sweeps_run_queue.csv \
  --output-csv artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/rollup_clean_cycle_top10.csv \
  --output-md artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/rollup_clean_cycle_top10.md \
  --lambda-dd 0.02 \
  --sort-mode score \
  --overwrite
```

Примечание: `lambda_dd` зависит от масштаба equity. Если меняется стартовый капитал (например, не $1000), шкала `canonical_max_drawdown_abs` меняется, поэтому `lambda_dd` нужно пересмотреть.

## Команды запуска на VPS (после подготовки; не выполнять локально)

Рекомендуемый способ heavy execution: `coint4/scripts/remote/run_server_job.sh` на VPS `85.198.90.128`.

Правила:
- По умолчанию `STOP_AFTER=1` (VPS выключится после job). Не ставить `STOP_AFTER=0` без явной причины.
- Если локальные изменения ещё не в origin, использовать `SYNC_UP=1` (rsync tracked файлов на VPS).
- Никогда не логировать/не печатать `SERVSPACE_API_KEY`.

Пример запуска baseline очереди (из `coint4/`):

```bash
export SERVSPACE_API_KEY="***"
export SERVER_ID="***"        # или SERVER_NAME="..."
export SERVER_IP="85.198.90.128"

# baseline batch
bash scripts/remote/run_server_job.sh \
  bash scripts/optimization/watch_wfa_queue.sh \
    --queue artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/baseline_run_queue.csv
```

Пример запуска sweeps очереди:

```bash
bash scripts/remote/run_server_job.sh \
  bash scripts/optimization/watch_wfa_queue.sh \
    --queue artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/sweeps_run_queue.csv
```

Быстрый sanity-check, что выполнение идёт на VPS (а не локально):

```bash
bash scripts/remote/run_server_job.sh \
  bash -lc 'echo RUN_HOST=$(hostname); uname -a; true'
```
