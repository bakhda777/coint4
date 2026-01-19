# Holdout diagnostics (2026-01-19, relaxed4)

## Контекст
- Holdout: `2024-05-01` → `2024-12-31`, `max_steps=5`.
- Прогоны:
  - `holdout_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03`
  - `holdout_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p5_ssd50000_kpss0p03`
- Референс WFA:
  - `stability_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p45_ssd50000_kpss0p03`
  - `stability_relaxed4_20260119_pv0p08_hurst0p65_hl0p02_60_corr0p5_ssd50000_kpss0p03`

## Шаги WFA (holdout, дневной Sharpe)
- corr0.45: median `-0.26`, min `-5.28`, worst step PnL `-89.03`, total PnL `-98.19`.
- corr0.5: median `-0.26`, min `-5.28`, worst step PnL `-69.53`, total PnL `-78.69`.

## Перекрытие пар с WFA
- corr0.45: holdout pairs `567`, WFA pairs `272`, intersection `4` (Jaccard `0.0048`), overlap PnL `-3.18`, non-overlap PnL `-148.53`.
- corr0.5: holdout pairs `560`, WFA pairs `270`, intersection `4` (Jaccard `0.00484`), overlap PnL `-3.18`, non-overlap PnL `-130.18`.

## Причины фильтрации (агрегировано)
- Holdout: pvalue, low_correlation, beta_out_of_range, kpss, hurst_too_high.
- WFA: low_correlation, pvalue, beta_out_of_range, kpss, hurst_too_high.
- Интерпретация: в holdout доминирует развал коинтеграции (pvalue), а перекрытие пар с WFA почти отсутствует.

## Выводы
- Расширение фильтров увеличило число пар, но потери в holdout концентрируются в непересекающихся парах.
- Требуется усилить устойчивость/согласованность отбора пар между WFA и holdout (например, увеличить `pair_stability_window_steps/min_steps` или фиксировать universe на базе train).

## Артефакты
- `coint4/results/holdout_relaxed4_20260119_step_metrics.csv`
- `coint4/results/holdout_relaxed4_20260119_pair_summary.csv`
- `coint4/results/holdout_relaxed4_20260119_asset_summary.csv`
- `coint4/results/holdout_relaxed4_20260119_pair_concentration.csv`
- `coint4/results/holdout_relaxed4_20260119_overlap_summary.json`
- `coint4/results/holdout_relaxed4_20260119_filter_summary.csv`
