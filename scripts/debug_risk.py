import sys
import os
import pandas as pd
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
    logger = logging.getLogger("debug_risk")
    
    # 1. Load Config
    config_path = "configs/main_2024_trae.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return
    
    cfg = load_config(config_path)
    print(f"✅ Loaded config from {config_path}")

    # 2. Create Dummy Data (Simulating a crash)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="15min")
    
    # Create a scenario where spread diverges massively
    # P1 stays flat, P2 crashes
    p1 = pd.Series([2000.0] * 100, index=dates)
    
    # P2 crashes from 100 to 50 starting at index 50
    # Spread = P1 - beta*P2. If P2 drops, spread increases.
    # We want spread to move +10 sigma quickly.
    
    import numpy as np
    np.random.seed(42)
    
    # Generate correlated random walk
    x = np.random.normal(0, 1, 100).cumsum() + 100
    y = x + np.random.normal(0, 0.1, 100) # Cointegrated
    
    # Introduce a break at index 60
    y[60:] += np.arange(40) * 2.0 # Massive divergence
    
    p1 = pd.Series(y, index=dates)
    p2 = pd.Series(x, index=dates)
    
    pair_data = pd.DataFrame({'P1': p1, 'P2': p2})
    
    # 3. Run Backtester with Strict Stop Loss
    print("\n🧪 Running Backtest with StopLoss=2.0 ...")
    
    bt = NumbaPairBacktester(
        pair_data=pair_data,
        rolling_window=20,
        z_threshold=1.0, # Easy entry
        z_exit=0.0,
        capital_at_risk=1000.0,
        stop_loss_multiplier=2.0, # Strict stop
        pair_name="TEST-CRASH"
    )
    
    # Inject config for validation
    bt.config = cfg.backtest
    bt.config.zscore_stop_loss = 2.0
    bt.config.max_zscore_entry = 10.0
    
    results = bt.run()
    
    # 4. Analyze Results
    trades = results['trades']
    print(f"\n📊 Trades found: {len(trades)}")
    
    stop_triggered = False
    for t in trades:
        print(f"  Trade: EntryZ={t.get('entry_z'):.2f}, ExitZ={t.get('exit_z'):.2f}, Reason={t.get('exit_reason')}")
        if t.get('exit_reason') == "StopLoss":
            stop_triggered = True
            
    if stop_triggered:
        print("\n✅ SUCCESS: Stop Loss triggered correctly!")
    else:
        print("\n❌ FAILURE: Stop Loss NOT triggered!")

if __name__ == "__main__":
    main()
