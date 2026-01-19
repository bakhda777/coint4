# Holdout diagnostics (2026-01-18)

Context:
- Holdout WFA runs: `holdout_20260118_baseline_z0p85_exit0p12_corr0p65_ssd25000` and `holdout_20260118_corr0p7_z0p85_exit0p12_ssd25000`.
- Configured dates: 2024-05-01 → 2024-12-31, but with `max_steps=5` and 30d test windows, the effective test range ends on 2024-09-28.

Key findings:
- Step-level: steps 1-4 are negative; step 2-3 drive the bulk of losses. Step 5 is ~flat.
- Pair overlap with shortlist baseline is tiny: 13/501 pairs (Jaccard ~1.7%). Overlapping pairs are net positive (+48.7), while non-overlap pairs are net negative (-370.2).
- Filter reasons shift: holdout is dominated by pvalue rejections, while shortlist is dominated by low correlation. This points to weaker cointegration in the holdout window.
- Worst assets by PnL contribution (baseline): GMT, CORE, GODS, MASK, ADA, STG, APT, STRK, NOT, BEL.
- Top losing pairs (baseline): GODSUSDT-MASKUSDT, COREUSDT-GMTUSDC, COREUSDT-SANDUSDT, COREUSDT-GMTUSDT, ETHWUSDT-TAIKOUSDT.

Artifacts (full details):
- Step metrics: `coint4/results/holdout_20260118_step_metrics.csv`
- Pair summary: `coint4/results/holdout_20260118_pair_summary.csv`
- Pair concentration (top winners/losers): `coint4/results/holdout_20260118_pair_concentration.csv`
- Asset concentration: `coint4/results/holdout_20260118_asset_summary.csv`
- Overlap summary: `coint4/results/holdout_20260118_overlap_summary.json`
- Filter summary: `coint4/results/holdout_20260118_filter_summary.csv`

Implications:
- Pair universe is unstable across regimes; shortlist pairs do not generalize to holdout.
- Cointegration quality degrades in holdout (pvalue filter dominates).
- Losses are spread but still concentrated enough to target a handful of pairs/assets.

Next actions:
- Add a stability constraint: require pairs to appear in >=N WFA steps or be present in both shortlist and holdout windows.
- Tighten filters for cointegration stability (lower pvalue, stricter half-life, lower max hurst).
- Consider restricting universe to higher-quality base assets with stable funding/volatility.
