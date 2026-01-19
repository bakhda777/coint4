# Holdout diagnostics (2026-01-19, relaxed5 w3m2)

## Контекст
- Holdout: `2024-05-01` → `2024-12-31`, фактически 4 шага (max_steps=5).
- Прогон: `holdout_relaxed5_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w3m2`.
- Референс WFA: `stability_relaxed5_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03_w3m2`.

## Шаги WFA (holdout, дневной Sharpe)
- Median `0.24`, min `-2.54`, worst step PnL `-141.32`, total PnL `-220.63`.

## Перекрытие пар с WFA
- Holdout pairs `1018`, WFA pairs `383`, intersection `16` (Jaccard `0.0116`).
- Overlap PnL `+14.32`, non-overlap PnL `-25.78`.

## Причины фильтрации (агрегировано)
- Holdout: pvalue, beta_out_of_range, low_correlation, kpss, hurst_too_high.
- WFA: low_correlation, pvalue, beta_out_of_range, kpss, hurst_too_high.
- Интерпретация: в holdout доминирует развал коинтеграции (pvalue), пересечение пар с WFA остаётся низким.

## Выводы
- Усиление стабильности (w3m2) повышает Sharpe на WFA, но holdout остаётся отрицательным.
- Нужен более стабильный universe/ограничение пар или изменение критериев отбора, чтобы увеличить пересечение WFA ↔ holdout.

## Артефакты
- `coint4/results/holdout_relaxed5_20260119_step_metrics.csv`
- `coint4/results/holdout_relaxed5_20260119_pair_summary.csv`
- `coint4/results/holdout_relaxed5_20260119_asset_summary.csv`
- `coint4/results/holdout_relaxed5_20260119_pair_concentration.csv`
- `coint4/results/holdout_relaxed5_20260119_overlap_summary.json`
- `coint4/results/holdout_relaxed5_20260119_filter_summary.csv`
