# Optuna Pilot Report
Generated: 2025-08-11T20:28:37.870804

## 📊 Study Overview
- **Study Name:** `coint2_portfolio_demo`
- **Objective:** PSR maximization
- **K-Folds:** 3
- **Pairs File:** `bench/pairs_portfolio.yaml`
- **Seed:** 42

## 📈 Trial Statistics
- **Total Trials:** 15
- **Completed:** 6
- **Pruned:** 9
- **Failed:** 0
- **Success Rate:** 40.0%

## 🎯 Performance Metrics
- **Best PSR:** 1.964
- **Median PSR:** 1.499
- **Top Decile Median:** 1.964
- **Top Decile Count:** 1

### Distribution Summary
- **Mean:** 1.402
- **Std:** 0.444
- **Min:** 0.685
- **25%:** 1.205
- **75%:** 1.618
- **Max:** 1.964

## 🏆 Top 10 Trials\n\n| Rank | Trial | PSR | Sharpe | DSR | Trades | Parameters |\n|------|-------|-----|--------|-----|--------|------------|\n| 1 | 12 | 1.964 | 2.294 | 1.742 | 1478 | z_th=1.51, z_ex=-0.04, w=74 |\n| 2 | 5 | 1.623 | 1.846 | 1.437 | 1423 | z_th=1.66, z_ex=-0.14, w=69 |\n| 3 | 4 | 1.606 | 1.887 | 1.438 | 2144 | z_th=1.54, z_ex=0.02, w=120 |\n| 4 | 7 | 1.392 | 1.612 | 1.228 | 1509 | z_th=1.98, z_ex=0.01, w=74 |\n| 5 | 3 | 1.143 | 1.328 | 1.003 | 2046 | z_th=1.56, z_ex=0.45, w=113 |\n| 6 | 0 | 0.685 | 0.790 | 0.591 | 1797 | z_th=2.25, z_ex=0.45, w=96 |\n\n## ⚙️ Parameter Ranges (Top Trials)\n\n- **zscore_threshold:** 1.510 - 2.249 (median: 1.610)\n- **zscore_exit:** -0.144 - 0.451 (median: 0.015)\n- **rolling_window:** 69.000 - 120.000 (median: 85.000)\n- **max_holding_days:** 188.000 - 295.000 (median: 259.000)\n- **seed:** 156019.000 - 981426.000 (median: 931461.500)\n\n## ✅ Sanity Checks\n\n- **PSR Check:** ✅ PASS (median top decile: 1.964)\n- **Trades Check:** ✅ PASS (avg trades: 1478)\n\n**Overall Status:** ✅ PILOT PASSED\n\n## 📁 Artifacts\n\n- **Best Trial Traces:** `artifacts/traces/optuna/trial_12_*`\n- **Trials CSV:** `artifacts/optuna/trials.csv`\n- **Best Params:** `artifacts/optuna/best_params.json`\n