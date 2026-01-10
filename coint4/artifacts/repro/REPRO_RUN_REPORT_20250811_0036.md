# Reproduction Run Report
Generated: 2025-08-11T00:36:04.460925
Source Manifest: artifacts/repro/RESULTS_MANIFEST.json

## 📊 Execution Summary
- **Total Steps:** 6
- **Successful:** 3
- **Failed:** 3
- **Success Rate:** 50.0%
- **Total Duration:** 25.5 seconds
- **Quick Mode:** ✅ Enabled

## 🔄 Step Results
| Step | Status | Duration | Description |
|------|--------|----------|-------------|
| 1 | ✅ | 1.1s | Проверить совместимость окружения |
| 2 | ✅ | 4.4s | Проверить целостность данных |
| 3 | ✅ | 9.2s | Воспроизвести анализ неопределённости |
| 4 | ❌ | 6.0s | Воспроизвести мониторинг дрейфа |
| 5 | ❌ | 0.6s | Воспроизвести paper week симуляцию |
| 6 | ❌ | 4.2s | Запустить CI gates для проверки качества |

## ❌ Failed Steps Detail

### Воспроизвести мониторинг дрейфа
**Command:** `python scripts/monitor_drift.py --config configs/drift_monitor.yaml`
**Error:** 

### Воспроизвести paper week симуляцию
**Command:** `python scripts/run_paper_week.py --pairs-file bench/pairs_portfolio.yaml --days 7`
**Error:** usage: run_paper_week.py [-h] [--pairs-file PAIRS_FILE]
                         [--portfolio-weights PORTFOLIO_WEIGHTS]
                         [--derisk-scale DERISK_SCALE] [--quiet]
run_paper_week

### Запустить CI gates для проверки качества
**Command:** `python scripts/ci_gates.py --config configs/ci_gates.yaml --verbose`
**Error:** 


## 📈 Metrics Comparison

- ✅ wfa_report: воспроизведён
- ✅ confidence_report: воспроизведён
- ✅ drift_dashboard: воспроизведён
- ✅ portfolio_report: воспроизведён

## 🎯 Recommendations

❌ **Poor reproduction** - significant issues detected. Review environment and data integrity.

⚡ **Quick mode was enabled** - some steps may have used reduced datasets.

## 🔍 Verification

To verify reproduction quality:
```bash
python scripts/ci_gates.py --config configs/ci_gates.yaml --verbose
```
