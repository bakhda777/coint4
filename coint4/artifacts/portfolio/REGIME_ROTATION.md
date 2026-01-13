> NOTE: Archived/legacy document. It may describe historical behavior and can be out of date. See `docs/` for current usage.

# Regime Rotation Log
Track of automatic portfolio rotations based on market regime detection.

---

## 2025-08-11 00:10:53 - Regime Rotation

### Regime Change
- **From**: N/A 
- **To**: low_vol
- **Confidence**: 60.00%

### Portfolio Update
- **Rebuild Triggered**: ❌ Failed
- **Reason**: Market regime shift detected

### Profile Applied (low_vol)
- **top_n**: 15
- **lambda_var**: 1.5
- **gamma_cost**: 0.8
- **max_weight_per_pair**: 0.18

### Next Steps
- Monitor performance under new regime profile
- Assess effectiveness of rotation after 3-7 days
- Consider manual override if regime detection is unstable

---

## 2025-08-11 00:11:16 - Regime Rotation

### Regime Change
- **From**: low_vol 
- **To**: high_vol
- **Confidence**: 80.00%

### Portfolio Update
- **Rebuild Triggered**: ✅ Yes
- **Reason**: Market regime shift detected

### Profile Applied (high_vol)
- **top_n**: 8
- **lambda_var**: 3.0
- **gamma_cost**: 1.5
- **max_weight_per_pair**: 0.12

### Next Steps
- Monitor performance under new regime profile
- Assess effectiveness of rotation after 3-7 days
- Consider manual override if regime detection is unstable

---
