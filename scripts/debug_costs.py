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
    logger = logging.getLogger("debug_costs")
    
    # 1. Load Config
    config_path = "configs/main_2024_trae.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return
    
    cfg = load_config(config_path)
    print(f"✅ Loaded config from {config_path}")
    
    # Force realistic costs
    cfg.backtest.enable_realistic_costs = True
    cfg.backtest.commission_rate_per_leg = 0.0004
    cfg.backtest.slippage_bps = 2.0
    
    print(f"🔧 Commission per leg: {cfg.backtest.commission_rate_per_leg}")
    print(f"🔧 Slippage bps: {cfg.backtest.slippage_bps}")

    # 2. Create Dummy Data (Flat price to isolate costs)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="15min")
    
    # P1 and P2 are stable, but we force a trade
    p1 = pd.Series([100.0] * 100, index=dates)
    p2 = pd.Series([100.0] * 100, index=dates)
    
    # Make P2 jump to trigger entry, then jump back
    # With mean=100, std=0 (initially), we need significant move
    # Rolling window = 5.
    # Let's make a gradual move so rolling stats adapt but Z spikes
    
    # Create distinct data
    # Sine waves with phase shift
    x = np.linspace(0, 100, 200)
    p1 = pd.Series(100 + 10 * np.sin(x), index=pd.date_range(start="2024-01-01", periods=200, freq="15min"))
    p2 = pd.Series(100 + 10 * np.sin(x + 0.1), index=p1.index) # Correlated but spread moves
    
    pair_data = pd.DataFrame({'P1': p1, 'P2': p2})
    
    # 3. Run Backtester
    print("\n🧪 Running Backtest to check costs...")
    
    # Relax entry threshold
    cfg.backtest.zscore_entry_threshold = 0.5
    cfg.backtest.zscore_threshold = 0.5
    
    bt = NumbaPairBacktester(
        pair_data=pair_data,
        rolling_window=20,
        z_threshold=0.5,
        z_exit=0.0,
        capital_at_risk=10000.0,
        pair_name="TEST-COSTS",
        config=cfg.backtest
    )
    
    # Override min_volatility to ensure Z-score isn't suppressed
    # Numba engine calculates min_volatility internally now, but we can try to influence it
    # Or just rely on standard calculation.
    
    results = bt.run()
    
    # 4. Analyze Results
    trades = results['trades']
    costs_series = results['costs']
    
    print(f"\n📊 Trades found: {len(trades)}")
    
    if len(trades) > 0:
        t = trades[0]
        print(f"  Trade 1: Gross={t.get('gross_pnl'):.4f}, Costs={t.get('costs'):.4f}, Net={t.get('net_pnl'):.4f}")
        
        if t.get('costs') > 0:
            print("\n✅ SUCCESS: Costs are being calculated!")
        else:
            print("\n❌ FAILURE: Costs are 0.00!")
            
        # Check total costs
        total_costs = costs_series.sum()
        print(f"  Total Costs (Series): {total_costs:.4f}")
    else:
        print("\n⚠️ No trades generated, cannot verify costs.")

if __name__ == "__main__":
    main()
