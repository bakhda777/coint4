import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coint2.utils.config import load_config
from coint2.pipeline.walk_forward_orchestrator import run_walk_forward

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("verify_integration")
    
    # 1. Load Config
    config_path = "configs/main_2024_trae.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return
    
    cfg = load_config(config_path)
    
    # 🔧 Override for speed
    cfg.walk_forward.start_date = "2024-01-01"
    cfg.walk_forward.end_date = "2024-01-10"
    cfg.walk_forward.training_period_days = 7
    cfg.walk_forward.testing_period_days = 2
    cfg.pair_selection.min_volume_usd_24h = 0 # Accept all volume
    cfg.pair_selection.liquidity_usd_daily = 0 # Accept all liquidity
    
    print("🚀 Starting Integration Verification...")
    try:
        # This requires data in 'data_downloaded'. 
        # Assuming data exists (as per previous logs).
        # If not, it will fail gracefully.
        results = run_walk_forward(cfg)
        
        if results['success']:
            print(f"\n✅ Verification Passed!")
            print(f"  Total Trades: {results['trade_count']}")
            print(f"  Total PnL: {results['total_pnl']:.2f}")
            print(f"  Sharpe: {results['sharpe_ratio_abs']:.4f}")
            
            if results['trade_count'] > 0:
                t = results['trades'][0]
                print(f"  Sample Trade: {t}")
        else:
            print(f"\n❌ Verification Failed: {results.get('error')}")
            
    except Exception as e:
        print(f"\n❌ Exception during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
