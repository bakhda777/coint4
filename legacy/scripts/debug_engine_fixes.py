import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coint2.utils.config import load_config
from coint2.engine.numba_engine import NumbaPairBacktester

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("debug_fixes")
    
    # 1. Load Config
    config_path = "configs/main_2024_trae.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return
    
    cfg = load_config(config_path)
    print(f"✅ Loaded config from {config_path}")
    
    # 🔧 SETUP CONFIG FOR TESTING
    cfg.backtest.enable_realistic_costs = True
    cfg.backtest.commission_rate_per_leg = 0.0004
    cfg.backtest.slippage_bps = 2.0
    cfg.backtest.min_position_hold_minutes = 60 # 4 bars (15 min each)
    cfg.backtest.anti_churn_cooldown_minutes = 30 # 2 bars
    cfg.backtest.zscore_entry_threshold = 2.0
    cfg.backtest.zscore_exit = 0.0
    
    # Disable filters for raw engine test
    cfg.backtest.market_regime_detection = False
    cfg.backtest.structural_break_protection = False
    cfg.backtest.adaptive_thresholds = False
    
    print(f"🔧 Test Params: Hold={cfg.backtest.min_position_hold_minutes}m, Cooldown={cfg.backtest.anti_churn_cooldown_minutes}m")

    # 2. Create Dummy Data
    dates = pd.date_range(start="2024-01-01", periods=200, freq="15min")
    
    # Case A: Normal Volatility, testing Hold Time
    # Sine wave with period ~50 bars
    x = np.linspace(0, 8*np.pi, 200)
    p1 = pd.Series(100 + 2 * np.sin(x), index=dates)
    p2 = pd.Series(100 + 2 * np.sin(x + 0.5), index=dates) # Spread will cycle
    
    pair_data = pd.DataFrame({'P1': p1, 'P2': p2})
    
    print("\n🧪 TEST 1: Hold Time Enforcement")
    bt = NumbaPairBacktester(
        pair_data=pair_data,
        rolling_window=20,
        z_threshold=1.0, # Lowered to ensure entry
        z_exit=0.0,
        capital_at_risk=10000.0,
        pair_name="TEST-HOLD",
        config=cfg.backtest
    )
    results = bt.run()
    
    # DEBUG: Print z-scores stats
    zs = results['results_df']['z_score']
    print(f"  Z-score stats: min={zs.min():.4f}, max={zs.max():.4f}, mean={zs.mean():.4f}")
    
    trades = results['trades']
    
    print(f"  Trades: {len(trades)}")
    for i, t in enumerate(trades):
        print(f"  Trade {i+1}: {t.get('side')} Entry={t.get('entry_time')} Exit={t.get('exit_time')} Hold={t.get('hold')} PnL={t.get('net_pnl'):.4f}")
        
        # Verify Hold Time (approximate string check)
        # Expected: '0 days 01:00:00' or more
        if '00:15:00' in str(t.get('hold')) or '00:30:00' in str(t.get('hold')):
             if t.get('exit_reason') != 'StopLoss':
                 print("  ❌ FAIL: Hold time too short for non-stoploss exit!")
             else:
                 print("  ✅ OK: Short hold allowed for StopLoss")
        else:
             print("  ✅ OK: Hold time respected")

    # Case B: Low Volatility (Extreme Z-score check)
    print("\n🧪 TEST 2: Low Volatility / Extreme Z-score Protection")
    # Increase noise to ensure variance > 1e-12
    p1_stable = pd.Series(100 + np.random.normal(0, 0.0001, 200), index=dates)
    p2_stable = pd.Series(100 + np.random.normal(0, 0.0001, 200), index=dates)
    
    # Inject a small jump that would be huge in sigma terms
    p2_stable.iloc[100:110] += 0.01 # Increased jump to ensure it's > 1e-5
    
    pair_data_stable = pd.DataFrame({'P1': p1_stable, 'P2': p2_stable})
    
    bt_stable = NumbaPairBacktester(
        pair_data=pair_data_stable,
        rolling_window=20,
        z_threshold=2.0,
        z_exit=0.0,
        capital_at_risk=10000.0,
        pair_name="TEST-LOW-VOL",
        config=cfg.backtest
    )
    results_stable = bt_stable.run()
    z_scores = results_stable['results_df']['z_score']
    
    max_z = np.nanmax(np.abs(z_scores))
    print(f"  Max Z-score observed: {max_z:.4f}")
    
    if max_z > 20.0:
        print("  ❌ FAIL: Z-score not clamped or min_volatility too small!")
    else:
        print("  ✅ SUCCESS: Z-score contained within limits.")

    # Check if min_volatility prevented 3000 sigma
    # We can inspect the 'std' column
    stds = results_stable['results_df']['std']
    valid_stds = stds[stds > 0]
    if len(valid_stds) > 0:
        min_std_observed = np.nanmin(valid_stds)
        print(f"  Min Std observed: {min_std_observed:.8f}")
    else:
        print("  ⚠️ No valid std observed")
    
    # We set min_volatility = 1e-5 in code.
    # If the reported Z-score uses clamped sigma, it should be fine.

if __name__ == "__main__":
    main()
