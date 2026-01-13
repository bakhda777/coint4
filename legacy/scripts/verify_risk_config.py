
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coint2.utils.config import load_config
import logging

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("verify_risk")
    
    config_path = "configs/main_2024_trae_fixed.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Config not found: {config_path}")
        return

    logger.info(f"Loading config from {config_path}")
    cfg = load_config(config_path)
    
    # Extract values
    pair_stop_loss_usd = getattr(cfg.backtest, 'pair_stop_loss_usd', None)
    pair_step_r_limit = getattr(cfg.backtest, 'pair_step_r_limit', None)
    risk_per_position_pct = getattr(cfg.backtest, 'risk_per_position_pct', None) or getattr(cfg.portfolio, 'risk_per_position_pct', 0.0)
    pnl_stop_loss_r_multiple = getattr(cfg.backtest, 'pnl_stop_loss_r_multiple', 0.0)
    
    print("\n" + "="*50)
    print("RISK CONFIGURATION CHECK")
    print("="*50)
    print(f"pair_stop_loss_usd       : {pair_stop_loss_usd} (Expected: 6.0)")
    print(f"pair_step_r_limit        : {pair_step_r_limit} (Expected: -3.0)")
    print(f"risk_per_position_pct    : {risk_per_position_pct}")
    print(f"pnl_stop_loss_r_multiple : {pnl_stop_loss_r_multiple}")
    print("="*50)
    
    # Validation
    if pair_stop_loss_usd == 6.0:
        print("✅ pair_stop_loss_usd is correct (6.0)")
    else:
        print(f"❌ pair_stop_loss_usd is INCORRECT! Got {pair_stop_loss_usd}")
        
    if pair_step_r_limit == -3.0:
        print("✅ pair_step_r_limit is correct (-3.0)")
    else:
        print(f"❌ pair_step_r_limit is INCORRECT! Got {pair_step_r_limit}")

if __name__ == "__main__":
    main()
