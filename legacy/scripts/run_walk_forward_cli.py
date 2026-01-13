
import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coint2.utils.config import load_config
from coint2.pipeline.walk_forward_orchestrator import run_walk_forward
import logging
from coint2.utils.logger import get_logger

# Setup logging
logger = get_logger("run_wf_cli")

def main():
    parser = argparse.ArgumentParser(description="Run Walk-Forward Analysis with CLI overrides")
    parser.add_argument("--config", default="configs/main_2024_trae_fixed.yaml", help="Path to config file")
    parser.add_argument("--start-date", help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--max-steps", type=int, help="Override max steps (0 for unlimited)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config {args.config} not found")
        return

    print(f"Loading config from {args.config}")
    cfg = load_config(args.config)
    
    overrides_used = False
    
    # Apply overrides
    if args.start_date:
        print(f"Overriding start_date: {cfg.walk_forward.start_date} -> {args.start_date}")
        cfg.walk_forward.start_date = args.start_date
        overrides_used = True
        
    if args.end_date:
        print(f"Overriding end_date: {cfg.walk_forward.end_date} -> {args.end_date}")
        cfg.walk_forward.end_date = args.end_date
        overrides_used = True
        
    if args.max_steps is not None:
        print(f"Overriding max_steps: {cfg.walk_forward.max_steps} -> {args.max_steps}")
        cfg.walk_forward.max_steps = args.max_steps
        overrides_used = True

    # Log effective range
    logger.info(f"WF_RANGE_EFFECTIVE "
                f"train_days={cfg.walk_forward.training_period_days}, "
                f"test_days={cfg.walk_forward.testing_period_days}, "
                f"max_steps={cfg.walk_forward.max_steps}, "
                f"overrides_used={overrides_used}")

    # Run
    result = run_walk_forward(cfg)
    
    # Save RESULT_JSON
    import json
    result_path = "wf_results.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
        
    print(f"\n✅ Walk-Forward completed. Results saved to {result_path}")
    
    # Validate Risk Config in Result
    try:
        saved_config = result.get('config', {})
        backtest_cfg = saved_config.get('backtest', {})
        pair_stop_loss_usd = backtest_cfg.get('pair_stop_loss_usd')
        
        print("\n[RESULT_JSON VALIDATION]")
        print(f"pair_stop_loss_usd: {pair_stop_loss_usd} (Expected: 6.0)")
        
        if pair_stop_loss_usd == 6.0:
            print("✅ RESULT_JSON contains correct pair_stop_loss_usd")
        else:
            print(f"❌ RESULT_JSON contains INCORRECT pair_stop_loss_usd: {pair_stop_loss_usd}")
            
    except Exception as e:
        print(f"❌ Validation Error: {e}")

if __name__ == "__main__":
    main()
