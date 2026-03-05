# coint4 / coint2 — заметки Claw (проект)

> Цель: быстро восстановить контекст по репозиторию, куда смотреть в коде, как запускать, где данные/артефакты, и где лежит «парная крипта» (cointegration pairs trading).

## 1) Что это
- Фреймворк для **парного трейдинга на крипте** на основе **коинтеграции** (scan пар, фиксированные бэктесты, walk-forward analysis, оптимизации/сетки параметров, UI).
- Python проект под Poetry. Имя пакета/CLI: **`coint2`**.

## 2) Структура репо (корень `/home/claudeuser/coint4`)
- `coint4/` — активный workspace (src/tests/configs/UI/Docker и т.п.).
- `docs/` — документация (в `coint4/` обычно есть удобная ссылка на неё).
- `legacy/` — архив/старый код и старые конфиги.
- `data/`, `outputs/`, `artifacts/`, `results/` — локальные данные/выгрузки/артефакты (многое игнорится в Git).

См. также: `README.md` в корне — описание layout + ссылки на quickstart/production checklist.

## 3) Где «точка входа» (CLI)
Poetry scripts (см. `coint4/pyproject.toml`):
- `coint2 = coint2.cli:main` — основной CLI.
- `coint2-optimize = coint2.optimiser.run_optimization:cli_main`
- `coint2-live = coint2.cli_live:main`
- `coint2-check-health = coint2.cli.check_coint_health:main`
- `coint2-build-universe = coint2.cli.build_universe:main`

Практически: внутри `coint4/` в README есть примеры вроде `./.venv/bin/coint2 ...`.

## 4) Ключевые части кода (где искать логику)
`coint4/src/coint2/`:
- `core/`
  - `fast_coint.py` — ускоренная коинтеграция/математика (важно для пар).
  - `pair_backtester.py` — бэктест пары.
  - `data_loader.py`, `data_prep.py` — загрузка/подготовка данных.
  - `numba_kernels*.py`, `numba_parity_v*.py` — ускорение/ядра на numba.
  - `portfolio.py`, `performance.py`, `sharpe.py`, `canonical_metrics.py` — метрики/портфель.
- `pipeline/`
  - `pair_scanner.py` — сканирование кандидатов/пар.
  - `filters.py`, `pair_ranking.py`, `walk_forward_orchestrator.py`, `cost_model.py`, `churn_control.py` — фильтры/ранжирование/WFA/издержки.
- `engine/` — движки бэктеста/оптимизированные реализации.
- `live/`
  - `bybit_client.py` — интеграция с Bybit.
  - `runner.py`, `paper_engine.py` — live/paper.
- `monitoring/` — safety/metrics.
- `utils/` — конфиги, детерминизм, dask utils, аудит данных и т.п.

`coint4/src/optimiser/`:
- инструменты оптимизации (в т.ч. Optuna), валидаторы (sharpe/lookahead), кэширование/менеджмент данных/метрик.

## 5) Конфиги / данные / артефакты
- Конфиги: `coint4/configs/` (их много; есть оверлеи под QA и «clean window»).
- Каноническая папка больших данных: `coint4/data_downloaded/` (игнорится, большие файлы).
- Артефакты WFA/оптимизаций: `coint4/artifacts/...` (в т.ч. `artifacts/wfa/aggregate/rollup/`).

Политика качества/окон данных упоминается в `coint4/README.md` (data_filters, чистое окно, исключения символов).

## 6) Типовые команды (шпаргалка)
См. `coint4/README.md` и `docs/quickstart.md`.

Примеры из `coint4/README.md`:
- `coint2 scan ...`
- `coint2 backtest ...`
- `coint2 walk-forward --config ...`
- UI (Streamlit): `streamlit run ui/app.py` с `PYTHONPATH=src`.

Тесты:
- `./.venv/bin/pytest -q`

## 7) Удалённые прогоны (важная политика)
В корневом `README.md` зафиксировано:
- **На этом сервере (146.103.41.248) тяжёлые прогоны не запускать**.
- WFA/оптимизации/долгие бэктесты — только на **85.198.90.128**.

Есть автоматизация для remote runs:
- `coint4/scripts/remote/run_server_job.sh` (поднимает сервер, ждёт SSH, тянет код, выполняет, синкает артефакты, выключает).

## 8) Последние прогоны / результаты (быстрый разбор)

Источник: `coint4/artifacts/wfa/aggregate/rollup/run_index.csv` (консолидированный индекс WFA/holdout запусков).

### 8.1 Что видно по последним holdout (OOS 2024-05-01 → 2025-06-30)
В окне `oos20240501_20250630` есть серия конфигов из `configs/evolution/20260226_evo_smoke_i154_20260302_025517/...`, которые показывают **очень высокий Sharpe (~5.6–6.0)** при умеренном DD (~-80…-95) и большом числе сделок (~1.2–1.3k).

Топ-кандидаты по этому окну (пример):
- `..._v018_evo_fdf6010c5c2f_oos20240501_20250630.yaml` — Sharpe 6.019, PnL 874.5, DD -89.2, trades 1292
- `..._v017_evo_b20ae04278b9_oos20240501_20250630.yaml` — Sharpe 5.912, PnL 947.3, DD -94.3, trades 1303
- `..._v005_evo_fcd7023fd60d_oos20240501_20250630.yaml` — Sharpe 5.774, PnL 917.1, DD -87.5, trades 1293
(и ещё ряд очень близких вариантов v001/v004/v015/v003/v012/...)

### 8.2 Важная проблема надёжности метрик на более ранних окнах OOS
Для тех же «base»-конфигов в окнах `oos20220601_20230430` и `oos20231001_20240930` в `strategy_metrics.csv` часто стоит:
- `total_pnl = 0.0`, `max_drawdown_abs = 0.0`, Sharpe = 0.0
при этом **`best_pair_pnl`/`worst_pair_pnl` и `total_costs` не нулевые** и `total_trades` > 0.

Пример:
`artifacts/wfa/runs/.../oos20220601_20230430/strategy_metrics.csv` показывает `total_pnl=0.0`, но `best_pair_pnl=+21.9`, `worst_pair_pnl=-20.3`, costs > 0.

Вывод: **сейчас нельзя честно сказать “этот конфиг стабильно хорош на всех доступных данных”**, пока не разберёмся, почему агрегированная PnL/Equity метрика нулевая на части holdout-окон.

### 8.3 Что нужно сделать, чтобы выбрать “параметры для лайва, в которых можно быть уверенным”
1) Выяснить причину нулевого `total_pnl/max_drawdown` при ненулевых per-pair PnL/costs (похоже на баг/пустой equity series/непроставленное поле).
2) Пересчитать/перегенерировать метрики для holdout окон и обновить rollup:
   - убедиться, что `strategy_metrics.csv` корректен
   - перегенерировать `artifacts/wfa/aggregate/rollup/run_index.csv` скриптом `scripts/optimization/build_run_index.py`
3) После фикса — ранжировать **по min/median Sharpe по нескольким OOS окнам**, с ограничениями по DD и стабильности PnL.

## 9) Что я сделаю дальше (если ок)
1) Локализую, где именно формируется `strategy_metrics.csv` (скрипт/модуль), и почему поля `total_pnl/max_drawdown_on_equity` пустые/нулевые.
2) Соберу таблицу “лучшие base-конфиги” по окну 2024-05→2025-06 + sanity-check на остальных окнах.

---
Обновляй/дополняй: этот файл — мои рабочие заметки для быстрого восстановления контекста.
