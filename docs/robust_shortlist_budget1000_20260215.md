# Robust shortlist ($1000)

Generated: 2026-02-15

Source: `coint4/artifacts/wfa/aggregate/rollup/run_index.csv`

## Gates (conservative defaults)
- Score: `worst_robust_sh` = min over windows of `min(holdout_sharpe, stress_sharpe)`
- DD gate: `worst_dd_pct` (max over windows of max(abs(dd_holdout), abs(dd_stress))) `<= 0.15`
- Sanity gates: `min_windows >= 3`, `min_trades >= 500`, `min_pairs >= 10` (min over windows)

Commands (run from `coint4/`):
```bash
PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py \
  --runs-dir artifacts/wfa/runs \
  --queue-dir artifacts/wfa/aggregate \
  --output-dir artifacts/wfa/aggregate/rollup

PYTHONPATH=src ./.venv/bin/python scripts/optimization/rank_multiwindow_robust_runs.py \
  --contains budget1000 --top 20 --min-windows 3 --min-trades 500 --min-pairs 10 --max-dd-pct 0.15
```

## Top variants (all budget1000)
| rank | worst_robust_sh | avg_robust_sh | worst_dd_pct | avg_dd_pct | windows | run_group | variant_id | sample_config |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | 3.530 | 4.388 | 0.132 | 0.089 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p91 | configs/budget_20260213_1000_dd_sprint08_stoplossusd_micro/holdout_prod_final_budget1000_oos20220601_20230430_risk0p006_slusd1p91.yaml |
| 2 | 3.530 | 4.388 | 0.132 | 0.089 | 3 | 20260214_budget1000_dd_sprint09_hurst_slusd1p91 | prod_final_budget1000_risk0p006_slusd1p91_max_hurst_exponent0p8 | configs/budget_20260214_1000_dd_sprint09_hurst_slusd1p91/holdout_prod_final_budget1000_oos20220601_20230430_risk0p006_slusd1p91_max_hurst_exponent0p8.yaml |
| 3 | 3.470 | 4.357 | 0.128 | 0.088 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p88 | configs/budget_20260213_1000_dd_sprint08_stoplossusd_micro/holdout_prod_final_budget1000_oos20220601_20230430_risk0p006_slusd1p88.yaml |
| 4 | 3.448 | 4.306 | 0.132 | 0.089 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p85_risk0p006 | configs/budget_20260213_1000_dd_sprint06_stoplossusd_fine/holdout_pruned168_oos20220601_20230430_slusd1p85_risk0p006.yaml |
| 5 | 3.448 | 4.306 | 0.132 | 0.089 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p85 | configs/budget_20260213_1000_dd_sprint08_stoplossusd_micro/holdout_prod_final_budget1000_oos20220601_20230430_risk0p006_slusd1p85.yaml |
| 6 | 3.323 | 4.254 | 0.137 | 0.087 | 3 | 20260213_budget1000_dd_sprint05_stoplossusd_refine | pruned168_slusd1p75_risk0p006 | configs/budget_20260213_1000_dd_sprint05_stoplossusd_refine/holdout_pruned168_slusd1p75_oos20220601_20230430_risk0p006.yaml |
| 7 | 3.323 | 4.254 | 0.137 | 0.087 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p75_risk0p006 | configs/budget_20260213_1000_dd_sprint06_stoplossusd_fine/holdout_pruned168_oos20220601_20230430_slusd1p75_risk0p006.yaml |
| 8 | 3.299 | 4.165 | 0.138 | 0.091 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p8_risk0p006 | configs/budget_20260213_1000_dd_sprint06_stoplossusd_fine/holdout_pruned168_oos20220601_20230430_slusd1p8_risk0p006.yaml |
| 9 | 3.179 | 4.200 | 0.137 | 0.086 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p65_risk0p006 | configs/budget_20260213_1000_dd_sprint06_stoplossusd_fine/holdout_pruned168_oos20220601_20230430_slusd1p65_risk0p006.yaml |
| 10 | 3.147 | 4.155 | 0.141 | 0.088 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p7_risk0p006 | configs/budget_20260213_1000_dd_sprint06_stoplossusd_fine/holdout_pruned168_oos20220601_20230430_slusd1p7_risk0p006.yaml |
| 11 | 2.864 | 4.115 | 0.108 | 0.073 | 3 | 20260213_budget1000_dd_sprint05_stoplossusd_refine | pruned168_slusd1p5_risk0p006 | configs/budget_20260213_1000_dd_sprint05_stoplossusd_refine/holdout_pruned168_slusd1p5_oos20220601_20230430_risk0p006.yaml |
| 12 | 2.297 | 3.879 | 0.142 | 0.091 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p97 | configs/budget_20260213_1000_dd_sprint08_stoplossusd_micro/holdout_prod_final_budget1000_oos20220601_20230430_risk0p006_slusd1p97.yaml |
| 13 | 2.211 | 3.517 | 0.143 | 0.101 | 3 | 20260213_budget1000_dd_sprint07_maxbeta_slusd2 | prod_final_budget1000_risk0p006_slusd2p0_max_beta20 | configs/budget_20260213_1000_dd_sprint07_maxbeta_slusd2/holdout_prod_final_budget1000_oos20220601_20230430_risk0p006_slusd2p0_max_beta20.yaml |
| 14 | 2.194 | 3.861 | 0.141 | 0.092 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p94 | configs/budget_20260213_1000_dd_sprint08_stoplossusd_micro/holdout_prod_final_budget1000_oos20220601_20230430_risk0p006_slusd1p94.yaml |
| 15 | 1.987 | 3.410 | 0.133 | 0.092 | 3 | 20260213_budget1000_dd_sprint07_maxbeta_slusd2 | prod_final_budget1000_risk0p006_slusd2p0_max_beta10 | configs/budget_20260213_1000_dd_sprint07_maxbeta_slusd2/holdout_prod_final_budget1000_oos20220601_20230430_risk0p006_slusd2p0_max_beta10.yaml |
| 16 | 1.691 | 3.492 | 0.148 | 0.099 | 3 | 20260213_budget1000_dd_sprint05_stoplossusd_refine | pruned168_slusd2p25_risk0p006 | configs/budget_20260213_1000_dd_sprint05_stoplossusd_refine/holdout_pruned168_slusd2p25_oos20220601_20230430_risk0p006.yaml |

## Focus: 2026-02-13 (budget1000)
| rank | worst_robust_sh | avg_robust_sh | worst_dd_pct | avg_dd_pct | windows | run_group | variant_id | sample_config |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | 3.530 | 4.388 | 0.132 | 0.089 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p91 | configs/budget_20260213_1000_dd_sprint08_stoplossusd_micro/holdout_prod_final_budget1000_oos20220601_20230430_risk0p006_slusd1p91.yaml |
| 2 | 3.470 | 4.357 | 0.128 | 0.088 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p88 | configs/budget_20260213_1000_dd_sprint08_stoplossusd_micro/holdout_prod_final_budget1000_oos20220601_20230430_risk0p006_slusd1p88.yaml |
| 3 | 3.448 | 4.306 | 0.132 | 0.089 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p85_risk0p006 | configs/budget_20260213_1000_dd_sprint06_stoplossusd_fine/holdout_pruned168_oos20220601_20230430_slusd1p85_risk0p006.yaml |
| 4 | 3.448 | 4.306 | 0.132 | 0.089 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p85 | configs/budget_20260213_1000_dd_sprint08_stoplossusd_micro/holdout_prod_final_budget1000_oos20220601_20230430_risk0p006_slusd1p85.yaml |
| 5 | 3.323 | 4.254 | 0.137 | 0.087 | 3 | 20260213_budget1000_dd_sprint05_stoplossusd_refine | pruned168_slusd1p75_risk0p006 | configs/budget_20260213_1000_dd_sprint05_stoplossusd_refine/holdout_pruned168_slusd1p75_oos20220601_20230430_risk0p006.yaml |
| 6 | 3.323 | 4.254 | 0.137 | 0.087 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p75_risk0p006 | configs/budget_20260213_1000_dd_sprint06_stoplossusd_fine/holdout_pruned168_oos20220601_20230430_slusd1p75_risk0p006.yaml |
| 7 | 3.299 | 4.165 | 0.138 | 0.091 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p8_risk0p006 | configs/budget_20260213_1000_dd_sprint06_stoplossusd_fine/holdout_pruned168_oos20220601_20230430_slusd1p8_risk0p006.yaml |
| 8 | 3.179 | 4.200 | 0.137 | 0.086 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p65_risk0p006 | configs/budget_20260213_1000_dd_sprint06_stoplossusd_fine/holdout_pruned168_oos20220601_20230430_slusd1p65_risk0p006.yaml |
| 9 | 3.147 | 4.155 | 0.141 | 0.088 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p7_risk0p006 | configs/budget_20260213_1000_dd_sprint06_stoplossusd_fine/holdout_pruned168_oos20220601_20230430_slusd1p7_risk0p006.yaml |
| 10 | 2.864 | 4.115 | 0.108 | 0.073 | 3 | 20260213_budget1000_dd_sprint05_stoplossusd_refine | pruned168_slusd1p5_risk0p006 | configs/budget_20260213_1000_dd_sprint05_stoplossusd_refine/holdout_pruned168_slusd1p5_oos20220601_20230430_risk0p006.yaml |
| 11 | 2.297 | 3.879 | 0.142 | 0.091 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p97 | configs/budget_20260213_1000_dd_sprint08_stoplossusd_micro/holdout_prod_final_budget1000_oos20220601_20230430_risk0p006_slusd1p97.yaml |
| 12 | 2.211 | 3.517 | 0.143 | 0.101 | 3 | 20260213_budget1000_dd_sprint07_maxbeta_slusd2 | prod_final_budget1000_risk0p006_slusd2p0_max_beta20 | configs/budget_20260213_1000_dd_sprint07_maxbeta_slusd2/holdout_prod_final_budget1000_oos20220601_20230430_risk0p006_slusd2p0_max_beta20.yaml |
| 13 | 2.194 | 3.861 | 0.141 | 0.092 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p94 | configs/budget_20260213_1000_dd_sprint08_stoplossusd_micro/holdout_prod_final_budget1000_oos20220601_20230430_risk0p006_slusd1p94.yaml |
| 14 | 1.987 | 3.410 | 0.133 | 0.092 | 3 | 20260213_budget1000_dd_sprint07_maxbeta_slusd2 | prod_final_budget1000_risk0p006_slusd2p0_max_beta10 | configs/budget_20260213_1000_dd_sprint07_maxbeta_slusd2/holdout_prod_final_budget1000_oos20220601_20230430_risk0p006_slusd2p0_max_beta10.yaml |
| 15 | 1.691 | 3.492 | 0.148 | 0.099 | 3 | 20260213_budget1000_dd_sprint05_stoplossusd_refine | pruned168_slusd2p25_risk0p006 | configs/budget_20260213_1000_dd_sprint05_stoplossusd_refine/holdout_pruned168_slusd2p25_oos20220601_20230430_risk0p006.yaml |

## Focus: 2026-02-14 (budget1000)
| rank | worst_robust_sh | avg_robust_sh | worst_dd_pct | avg_dd_pct | windows | run_group | variant_id | sample_config |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | 3.530 | 4.388 | 0.132 | 0.089 | 3 | 20260214_budget1000_dd_sprint09_hurst_slusd1p91 | prod_final_budget1000_risk0p006_slusd1p91_max_hurst_exponent0p8 | configs/budget_20260214_1000_dd_sprint09_hurst_slusd1p91/holdout_prod_final_budget1000_oos20220601_20230430_risk0p006_slusd1p91_max_hurst_exponent0p8.yaml |
