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
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("verify_pnl")
    
    # 1. Config
    config_path = "configs/main_2024_trae.yaml"
    cfg = load_config(config_path)
    
    # Force parameters for verification
    cfg.backtest.enable_realistic_costs = False
    cfg.backtest.commission_pct = 0.001 # 0.1% for visibility
    cfg.backtest.slippage_pct = 0.0
    cfg.backtest.zscore_entry_threshold = 1.0
    cfg.backtest.zscore_exit = 0.0
    
    # 2. Synthetic Data
    # Price ~ $100
    dates = pd.date_range(start="2024-01-01", periods=200, freq="15min")
    x = np.linspace(0, 8*np.pi, 200)
    p1 = pd.Series(100 + 5 * np.sin(x), index=dates) 
    p2 = pd.Series(100 + 5 * np.sin(x + 0.5), index=dates) # Phase shift
    
    pair_data = pd.DataFrame({'P1': p1, 'P2': p2})
    
    capital = 10000.0
    
    print("\n🧪 STARTING PnL VERIFICATION")
    print(f"Capital: ${capital}")
    print(f"Commission: {cfg.backtest.commission_pct}")
    
    bt = NumbaPairBacktester(
        pair_data=pair_data,
        rolling_window=20,
        z_threshold=1.0,
        z_exit=0.0,
        capital_at_risk=capital,
        pair_name="VERIFY_PNL",
        config=cfg.backtest
    )
    
    results = bt.run()
    
    if results is None:
        print("❌ Backtest failed to run")
        return

    df = results['results_df']
    trades = results['trades']
    
    print(f"\nTrades found: {len(trades)}")
    
    if len(trades) == 0:
        print("DEBUG: Z-scores stats:")
        print(df['z_score'].describe())
        print("DEBUG: Positions stats:")
        print(df['position'].value_counts())
        
    # Manual Verification of Trade 1
    if len(trades) > 0:
        t = trades[0]
        print(f"\n--- Trade 1 Analysis ---")
        print(f"Type: {t['side']}")
        print(f"Entry: {t['entry_time']} @ Z={t['entry_z']:.2f}")
        print(f"Exit: {t['exit_time']} @ Z={t['exit_z']:.2f}")
        print(f"Engine Net PnL: {t['net_pnl']:.4f}")
        print(f"Engine Costs: {t['costs']:.4f}")
        print(f"Engine Gross PnL: {t['gross_pnl']:.4f}")
        
        # Manual Calc
        # 1. Get Entry/Exit Prices
        entry_idx = df.index.get_loc(t['entry_time'])
        exit_idx = df.index.get_loc(t['exit_time'])
        
        # Positions
        # Engine uses position size. 
        # Check scaled position size
        pos_series = df['position']
        # Position during trade (take middle)
        pos_val = pos_series.iloc[entry_idx] 
        print(f"Scaled Position Size: {pos_val:.4f}")
        
        # Scaling Factor
        avg_price = pair_data.iloc[:, 0].mean() # Using P1 as approx
        # Engine uses np.nanmean(y) where y is P1
        scaling_factor = capital / avg_price
        print(f"Scaling Factor (Capital/Price): {scaling_factor:.4f}")
        
        # Expected Raw Position (Kernel)
        raw_pos = 1.0 if t['side'] == 'LONG' else -1.0
        
        # Verify Position Scaling
        if abs(pos_val - (raw_pos * scaling_factor)) > 0.1:
             print(f"⚠️ Position scaling mismatch! Engine: {pos_val}, Manual: {raw_pos * scaling_factor}")
        else:
             print("✅ Position scaling correct")
             
        # Calculate Raw PnL (Spread Change)
        # CORRECT FORMULA: PnL = Position * ((Y[i] - Y[i-1]) - beta[i-1] * (X[i] - X[i-1]))
        
        # Get Data
        y_series = df['spread'] + df['beta'] * pair_data.iloc[:, 1] # Recover Y? No, df['spread'] is result.
        # Better use pair_data directly
        y_series = pair_data.iloc[:, 0]
        x_series = pair_data.iloc[:, 1]
        beta_series = df['beta']
        
        manual_gross_pnl = 0.0
        for i in range(entry_idx + 1, exit_idx + 1):
            y_curr = y_series.iloc[i]
            y_prev = y_series.iloc[i-1]
            x_curr = x_series.iloc[i]
            x_prev = x_series.iloc[i-1]
            
            b_prev = beta_series.iloc[i-1]
            
            # Position is held from i-1
            pos_held = pos_series.iloc[i-1]
            
            # PnL
            step_pnl = pos_held * ((y_curr - y_prev) - b_prev * (x_curr - x_prev))
            manual_gross_pnl += step_pnl
            
        print(f"Manual Gross PnL (sum of daily): {manual_gross_pnl:.4f}")
        
        if abs(manual_gross_pnl - t['gross_pnl']) < 1.0:
             print("✅ Gross PnL matches")
        else:
             print(f"❌ Gross PnL Mismatch! Diff: {manual_gross_pnl - t['gross_pnl']:.4f}")

        # Calculate Costs
        # Cost = Trade Size * Rate * Price * Scaling
        # Trade Size = abs(New - Old)
        # Entry Cost
        # Scaled Trade Size = abs(pos_series[entry_idx] - pos_series[entry_idx-1])
        # But wait, scaling is applied to cost_series in engine.
        # Kernel calculates cost = trade_size_raw * pct.
        # Engine scales: cost_scaled = cost_raw * scaling * price.
        
        # Entry
        trade_size_raw = 1.0 # 0 -> 1
        entry_cost_raw = trade_size_raw * cfg.backtest.commission_pct
        entry_cost_scaled = entry_cost_raw * scaling_factor * avg_price
        
        # Exit
        trade_size_raw = 1.0 # 1 -> 0
        exit_cost_raw = trade_size_raw * cfg.backtest.commission_pct
        exit_cost_scaled = exit_cost_raw * scaling_factor * avg_price
        
        manual_total_costs = entry_cost_scaled + exit_cost_scaled
        print(f"Manual Costs: {manual_total_costs:.4f}")
        
        if abs(manual_total_costs - t['costs']) < 0.1:
             print("✅ Costs match")
        else:
             print(f"❌ Costs Mismatch! Diff: {manual_total_costs - t['costs']:.4f}")
             
        manual_net = manual_gross_pnl - manual_total_costs
        print(f"Manual Net PnL: {manual_net:.4f}")
        
if __name__ == "__main__":
    main()
