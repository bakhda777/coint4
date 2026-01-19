# Holdout diagnostics (2026-01-19, relaxed6 w3m2)

## Контекст
- Holdout: `2024-05-01` → `2024-12-31`, 5 шагов (max_steps=5).
- Прогон: `holdout_relaxed6_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2`.
- Референс WFA: `stability_relaxed6_20260119_pv0p12_hurst0p7_hl0p02_60_corr0p45_ssd50000_kpss0p05_w3m2`.

## Шаги WFA (holdout, дневной Sharpe)
- Median `-2.20`, min `-3.58`, worst step PnL `-178.94`, total PnL `-267.02`.

## Перекрытие пар с WFA
- Holdout pairs `779`, WFA pairs `303`, intersection `6` (Jaccard `0.0056`).
- Overlap PnL `+25.07`, non-overlap PnL `-56.34`.

## Причины фильтрации (агрегировано)
- Holdout: pvalue, beta_out_of_range, low_correlation, kpss, hurst_too_high.
- WFA: low_correlation, pvalue, beta_out_of_range, kpss, hurst_too_high.
- Интерпретация: перекрытие пар остаётся минимальным, а убыток формируется вне пересечения.

## Выводы
- Ослабление pvalue/kpss/hurst сохраняет высокий Sharpe на WFA, но holdout резко отрицательный.
- Требуется смена подхода к universe/фильтрам для 2024H2 или альтернативная модель отбора.

## Артефакты
- `coint4/results/holdout_relaxed6_20260119_step_metrics.csv`
- `coint4/results/holdout_relaxed6_20260119_pair_summary.csv`
- `coint4/results/holdout_relaxed6_20260119_asset_summary.csv`
- `coint4/results/holdout_relaxed6_20260119_pair_concentration.csv`
- `coint4/results/holdout_relaxed6_20260119_overlap_summary.json`
- `coint4/results/holdout_relaxed6_20260119_filter_summary.csv`
