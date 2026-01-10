import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coint2.utils.config import load_config
from coint2.core.data_loader import DataHandler
from coint2.engine.numba_engine import NumbaPairBacktester

import logging
import coint2

def main():
    logging.basicConfig(level=logging.DEBUG)
    print("🚀 Starting Debug Single Pair...")
    
    # 1. Load Config
    config_path = "configs/main_2024_trae.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return
    
    cfg = load_config(config_path)
    # Override for single pair debug if needed
    # cfg.backtest.zscore_entry_threshold = 1.5
    # cfg.backtest.zscore_threshold = 1.5
    # cfg.backtest.market_regime_detection = False
    # cfg.backtest.structural_break_protection = False
    # cfg.backtest.enable_funding_time_filter = False
    # cfg.backtest.enable_macro_event_filter = False
    # cfg.backtest.max_zscore_entry = 10.0 # Relax for debug
    # cfg.backtest.stop_loss_multiplier = 10.0 # Relax for debug
    
    print(f"✅ Loaded config from {config_path}")
    
    # 2. Load Data
    print(f"📥 Loading data from {cfg.data_dir}...")
    data_handler = DataHandler(cfg)
    
    # Load a bit more data than needed for rolling window
    start_date = pd.Timestamp("2024-01-15")
    end_date = pd.Timestamp("2024-01-22")
    lookback_days = 30
    
    price_df = data_handler.load_all_data_for_period(
        lookback_days=(end_date - start_date).days + lookback_days,
        end_date=end_date
    )
    
    if price_df.empty:
        print("❌ No data loaded!")
        return

    # 3. Select Pair
    s1 = "ETHDAI"
    s2 = "METHUSDT"
    pair_name = f"{s1}-{s2}"
    
    if s1 not in price_df.columns or s2 not in price_df.columns:
        print(f"❌ Symbols {s1} or {s2} not found in data columns: {price_df.columns.tolist()}")
        return

    pair_data = price_df[[s1, s2]].loc[start_date:end_date].dropna()
    print(f"✅ Selected pair {pair_name}, data shape: {pair_data.shape}")
    
    if pair_data.empty:
        print("❌ Pair data is empty after filtering/dropping NaNs")
        return

    # 4. Run Backtest
    print("🔄 Running NumbaPairBacktester...")
    bt = NumbaPairBacktester(
        pair_data=pair_data,
        rolling_window=cfg.backtest.rolling_window,
        z_threshold=cfg.backtest.zscore_threshold,
        z_exit=cfg.backtest.zscore_exit,
        capital_at_risk=1000.0,
        stop_loss_multiplier=cfg.backtest.stop_loss_multiplier,
        time_stop_multiplier=cfg.backtest.time_stop_multiplier,
        max_position_size_pct=1.0,
        pair_name=pair_name,
        commission_pct=cfg.backtest.commission_pct,
        slippage_pct=cfg.backtest.slippage_pct,
        config=cfg.backtest # Pass the full config
    )
    
    results = bt.run()
    
    if results is None:
        print("❌ Backtest returned None!")
        return

    # 5. Analyze Results
    results_df = results['results_df']
    results_df['price1'] = pair_data[s1]
    results_df['price2'] = pair_data[s2]
    
    # Save to CSV
    output_csv = "outputs/debug_results.csv"
    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv(output_csv)
    print(f"💾 Saved full results to {output_csv}")
    
    # Print Analysis
    print("\n🔍 Analysis:")
    print(f"Total PnL: {results_df['pnl'].sum():.4f}")
    print(f"Trades: {len(results['trades'])}")
    
    # Check for anomalies
    nan_z = results_df['z_score'].isna().sum()
    inf_z = np.isinf(results_df['z_score']).sum()
    max_z = results_df['z_score'].abs().max()
    
    print(f"NaN Z-scores: {nan_z}")
    print(f"Inf Z-scores: {inf_z}")
    print(f"Max |Z-score|: {max_z}")
    
    # Print extreme rows
    extreme_mask = results_df['z_score'].abs() > 10
    if extreme_mask.any():
        print("\n⚠️ Extreme Z-scores detected:")
        print(results_df[extreme_mask][['price1', 'price2', 'spread', 'mean', 'std', 'z_score']].head())
        
    # Print first few rows
    print("\n📋 First 5 rows:")
    print(results_df[['price1', 'price2', 'spread', 'mean', 'std', 'z_score', 'position', 'pnl']].head())

if __name__ == "__main__":
    main()
