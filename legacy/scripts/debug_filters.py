import sys
import os
import pandas as pd
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coint2.utils.config import load_config
from coint2.core.data_loader import DataHandler
from coint2.pipeline.filters import enhanced_pair_screening

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("debug_filters")
    
    # 1. Load Config
    config_path = "configs/main_2024_trae.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return
    
    cfg = load_config(config_path)
    print(f"✅ Loaded config from {config_path}")
    
    # Print what values we expect
    print("\n📊 Expected Config Values:")
    pvalue_threshold = getattr(cfg.pair_selection, 'coint_pvalue_threshold', 0.05)
    print(f"  coint_pvalue_threshold: {pvalue_threshold} (from pair_selection)")
    
    max_half_life_days = getattr(cfg.pair_selection, 'max_half_life_days',
                                getattr(cfg.filter_params, 'max_half_life_days', 14))
    print(f"  max_half_life_days: {max_half_life_days} (from pair_selection/filter_params)")
    
    min_daily_volume_usd = getattr(cfg.pair_selection, 'liquidity_usd_daily', 
                                  getattr(cfg.pair_selection, 'min_volume_usd_24h', 50000.0))
    if min_daily_volume_usd is None: min_daily_volume_usd = 50000.0
    print(f"  liquidity_usd_daily: {min_daily_volume_usd} (from pair_selection)")
    
    max_hurst_exponent = getattr(cfg.pair_selection, 'max_hurst_exponent',
                                getattr(cfg.filter_params, 'max_hurst_exponent', 0.5))
    if max_hurst_exponent is None: max_hurst_exponent = 0.5
    print(f"  max_hurst_exponent: {max_hurst_exponent} (from pair_selection/filter_params)")

    # 2. Load Data
    print(f"\n📥 Loading data from {cfg.data_dir}...")
    data_handler = DataHandler(cfg)
    
    # Load data for recent period
    end_date = pd.Timestamp("2024-01-20") # Use a date where we have data
    lookback_days = 30
    
    price_df = data_handler.load_all_data_for_period(
        lookback_days=lookback_days,
        end_date=end_date
    )
    
    if price_df.empty:
        print("❌ No data loaded!")
        return
        
    print(f"✅ Loaded data shape: {price_df.shape}")

    # 3. Define candidates
    candidates = [
        ("ETHDAI", "METHUSDT"),
        ("ORDIUSDT", "SOLUSDT")
    ]
    
    # Check if symbols exist
    valid_candidates = []
    for s1, s2 in candidates:
        if s1 in price_df.columns and s2 in price_df.columns:
            valid_candidates.append((s1, s2))
        else:
            print(f"⚠️ Missing data for {s1}-{s2}")
            
    if not valid_candidates:
        print("❌ No valid candidates found in loaded data")
        return

    # 4. Run Screening
    print(f"\n🔍 Running enhanced_pair_screening on {len(valid_candidates)} pairs...")
    
    screened = enhanced_pair_screening(
        valid_candidates,
        price_df,
        pvalue_threshold=pvalue_threshold,
        max_half_life_bars=int(max_half_life_days * 96),
        min_daily_volume_usd=min_daily_volume_usd,
        max_hurst_exponent=max_hurst_exponent,
        save_filter_reasons=True
    )
    
    print(f"\n✅ Passed pairs: {len(screened)}")
    for item in screened:
        print(f"  Passed: {item[0]}-{item[1]}")

if __name__ == "__main__":
    main()
