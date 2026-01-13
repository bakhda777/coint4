
import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coint2.utils.config import load_config
import logging

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("final_validate")
    
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
    
    # Filter values
    # Need to check if pair_selection still has them or if they are gone
    # We removed them from logic, but they might still be in YAML/Config object if not removed from model
    # Actually we removed them from YAML but the Model might still have defaults.
    # The goal was to unify.
    
    print("\n" + "="*50)
    print("FINAL CONFIGURATION VALIDATION")
    print("="*50)
    print(f"pair_stop_loss_usd       : {pair_stop_loss_usd} (Expected: 6.0)")
    print(f"pair_step_r_limit        : {pair_step_r_limit} (Expected: -3.0)")
    print(f"risk_per_position_pct    : {risk_per_position_pct}")
    print(f"pnl_stop_loss_r_multiple : {pnl_stop_loss_r_multiple}")
    print("="*50)
    
    # Validation
    errors = []
    if pair_stop_loss_usd != 6.0:
        errors.append(f"pair_stop_loss_usd is {pair_stop_loss_usd}, expected 6.0")
        
    if pair_step_r_limit != -3.0:
        errors.append(f"pair_step_r_limit is {pair_step_r_limit}, expected -3.0")
        
    if errors:
        print("❌ VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("✅ VALIDATION SUCCESS")
        
    # Generate RESULT_JSON dummy
    result_json = {
        "config": {
            "risk": {
                "pair_stop_loss_usd": pair_stop_loss_usd,
                "pair_step_r_limit": pair_step_r_limit
            }
        }
    }
    print("\nGenerated RESULT_JSON snippet:")
    print(json.dumps(result_json, indent=2))

if __name__ == "__main__":
    main()
