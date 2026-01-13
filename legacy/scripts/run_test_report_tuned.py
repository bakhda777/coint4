
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coint2.utils.config import load_config
from coint2.pipeline.walk_forward_orchestrator import run_walk_forward
import logging
from coint2.utils.logger import get_logger
import pandas as pd
import numpy as np

# Setup logging
logger = get_logger("test_run")

def main():
    config_path = "configs/main_2024_trae_fixed.yaml"
    print(f"Loading config from {config_path}")
    
    try:
        cfg = load_config(config_path)
        
        # Print WF_CONFIG style info
        print(f"WF_CONFIG: stop_R={cfg.backtest.pnl_stop_loss_r_multiple} pair_step_r_multiple={cfg.risk_limits.pair_step_r_multiple}")
        
        print("Starting Walk-Forward...")
        results = run_walk_forward(cfg)
        
        if results['success']:
            print("SUCCESS")
            print(f"Total PnL: {results['total_pnl']:.2f}")
            print(f"Sharpe: {results['sharpe_ratio_abs']:.4f}")
            print(f"Trades: {results['trade_count']}")
            
            # Analysis
            trades = results.get('trades_log', [])
            if trades:
                # Convert to DF for easier analysis
                df = pd.DataFrame(trades)
                if not df.empty and 'final_pnl_r' in df.columns:
                    max_loss_r = df['final_pnl_r'].min()
                    print(f"Max Loss R (Single Trade): {max_loss_r:.4f}R")
                    
                    # Analyze Exit Reasons
                    pnl_stop_r_count = len(df[df['exit_reason'] == 'PnLStopHardR'])
                    pnl_stop_usd_count = len(df[df['exit_reason'] == 'PnLStopHardUSD'])
                    
                    avg_loss_r_stop = df[df['exit_reason'] == 'PnLStopHardR']['net_pnl'].mean() if pnl_stop_r_count > 0 else 0.0
                    avg_loss_usd_stop = df[df['exit_reason'] == 'PnLStopHardUSD']['net_pnl'].mean() if pnl_stop_usd_count > 0 else 0.0
                    
                    print(f"PnLStopHardR Count: {pnl_stop_r_count}, Avg Loss: ${avg_loss_r_stop:.2f}")
                    print(f"PnLStopHardUSD Count: {pnl_stop_usd_count}, Avg Loss: ${avg_loss_usd_stop:.2f}")
                    
                    # Calculate comm_to_gross_ratio
                    gross_pnl = df['gross_pnl'].sum()
                    total_fees = df['commission'].sum() + df['slippage'].sum() if 'slippage' in df.columns else df['commission'].sum()
                    net_pnl = df['net_pnl'].sum()
                    comm_ratio = total_fees / gross_pnl if gross_pnl > 0 else 0.0
                    
                    print(f"Gross PnL: {gross_pnl:.2f}")
                    print(f"Total Fees: {total_fees:.2f}")
                    print(f"Net PnL: {net_pnl:.2f}")
                    print(f"Comm/Gross Ratio: {comm_ratio:.2f}")
                    
                    # Step Analysis
                    # We can approximate steps by date or if 'step' column exists (it usually doesn't in trades_log list unless added)
                    # But we can see it in logs.
                    
                    # Pair Analysis
                    pair_stats = df.groupby('pair').agg({
                        'final_pnl_r': 'sum',
                        'net_pnl': 'sum',
                        'exit_reason': lambda x: (x == 'PnLStopHardR').any()
                    })
                    print("\nPair Analysis:")
                    print(pair_stats)
                    
                    min_cum_pnl_r = pair_stats['final_pnl_r'].min()
                    print(f"\nMin CumPnL R per Pair: {min_cum_pnl_r:.4f}R")
                    
        else:
            print(f"FAILED: {results.get('error')}")
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
