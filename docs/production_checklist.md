# Чек-лист production-запуска

Для прод-пайплайна используйте `configs/main_2024.yaml` (поверх `configs/data_window_clean.yaml`). `configs/prod.yaml` предназначен для live/paper и не используется в scan/backtest/WFA.

1. Обновить данные в `coint4/data_downloaded/` и убедиться, что структура помесячная (`year=YYYY/month=MM`).
2. Зафиксировать `data_filters.clean_window` и список `data_filters.exclude_symbols` (сейчас пустой) в `configs/main_2024.yaml` / `configs/data_window_clean.yaml`.
3. Для WFA убедиться, что `walk_forward.start_date` ≥ `clean_window.start_date + training_period_days` (по умолчанию 60 дней → `2022-04-30`).
4. Для fixed‑backtest использовать `--period-start/--period-end` внутри clean window.
5. Запустить валидацию данных:
   - базовая: `PYTHONPATH=src ./.venv/bin/python scripts/validate_data_dump.py --data-root data_downloaded --mode monthly --config configs/main_2024.yaml`
   - строгая: `PYTHONPATH=src ./.venv/bin/python scripts/validate_data_dump.py --data-root data_downloaded --mode monthly --config configs/data_quality_strict.yaml`
6. Проверить отчеты `coint4/outputs/data_quality_report*.csv` и при необходимости обновить `data_filters.clean_window` / `data_filters.exclude_symbols`.
7. Подтвердить модель издержек: агрегированная (`commission_pct` + `slippage_pct`), `enable_realistic_costs: false`.
8. Проверить лимит `walk_forward.max_steps` (по умолчанию 5, для большего согласовать).
9. Сгенерировать пары: `./.venv/bin/coint2 scan --config configs/criteria_relaxed.yaml --base-config configs/main_2024.yaml --output-dir bench`.
10. Запустить fixed backtest на выбранных парах и периоде, сохранить метрики в `outputs/`.
11. Запустить walk-forward `./.venv/bin/coint2 walk-forward --config configs/main_2024.yaml` и проверить `results/`.
   Для ускоренной проверки можно использовать `configs/main_2024_wfa_balanced.yaml`.
12. Сверить метрики (P&L, Sharpe, max drawdown) с предыдущей базовой линией.
13. Проверить повторяемость (повторить fixed backtest и сравнить метрики).
14. Прогнать тесты: `./.venv/bin/pytest -q`.
15. Обновить документацию и сохранить конфиги/артефакты в репозитории.

## Последний прогон (2026-01-14)

- Universe (clean window, top-200): `bench/clean_window_20260114_top200_step3/` (tested: 5253, passing: 598, selected: 200).
- Fixed backtest: `outputs/fixed_run_clean_window_top200_20260114_step3/` (total_pnl: -243.99, sharpe_ratio: -0.1758).
- Fixed repeat: `outputs/fixed_run_clean_window_top200_20260114_step3_repeat/` (diff = 0).
- WFA:
  - main: `coint4/artifacts/wfa/runs/20260114_093244_main_2024_wfa_step5/`
  - main repeat: `coint4/artifacts/wfa/runs/20260114_095105_main_2024_wfa_step5_repeat/`
  - balanced: `coint4/artifacts/wfa/runs/20260114_072317_balanced_2024_wfa/`
  - balanced repeat: `coint4/artifacts/wfa/runs/20260114_073405_balanced_2024_wfa_repeat/`
- Детерминизм: fixed и main/balanced WFA повторены без отличий в метриках.
- Итоговый отчет: `docs/final_report_20260114.md`.
- Fast iteration (smoke): `bench/fast_iter_20260114_top50/`, `bench/fast_iter_20260114_top100/`.
- Fast fixed (1 месяц): `outputs/fixed_run_fast_iter_20260114_top50/` + repeat `outputs/fixed_run_fast_iter_20260114_top50_repeat/`.
- Fast fixed (Q3): `outputs/fixed_run_fast_iter_20260114_top50_q3/`, `outputs/fixed_run_fast_iter_20260114_top100/`.
- WFA main refresh (legacy): `coint4/artifacts/wfa/runs/20260114_081835_main_2024_wfa_refresh/`.
- WFA smoke: `coint4/artifacts/wfa/runs/20260114_075638_smoke_fast20_wfa/` + repeat `coint4/artifacts/wfa/runs/20260114_075758_smoke_fast20_wfa_repeat/`.
