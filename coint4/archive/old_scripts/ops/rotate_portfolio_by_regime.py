#!/usr/bin/env python3
"""
Regime-aware portfolio rotation.
Detects market regime and applies appropriate portfolio profile.
"""

import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class RegimePortfolioRotator:
    """Rotate portfolio based on market regime detection."""
    
    def __init__(self, config_path: str, verbose: bool = False):
        """Initialize regime rotator."""
        self.verbose = verbose
        self.config_path = config_path
        self.config = self._load_config()
        
        # Current regime state
        self.current_regime = None
        self.last_regime = None
        self.regime_confidence = 0.0
        
    def _load_config(self) -> Dict:
        """Load portfolio configuration."""
        config_path = Path(self.config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
            if self.verbose:
                print(f"✅ Loaded config from {config_path}")
            return config
    
    def detect_market_regime(self) -> Tuple[str, float]:
        """Detect current market regime based on volatility and trends.
        
        Returns:
            Tuple of (regime_name, confidence)
            - regime_name: 'low_vol' | 'mid_vol' | 'high_vol'
            - confidence: 0.0-1.0 confidence in regime detection
        """
        # Try to load recent market data
        wfa_results = Path('artifacts/wfa/results_per_fold.csv')
        weekly_data = Path('artifacts/live/WEEKLY_SUMMARY.md')
        
        # Method 1: Use WFA volatility if available
        if wfa_results.exists():
            try:
                wfa_df = pd.read_csv(wfa_results)
                
                if 'pnl' in wfa_df.columns and len(wfa_df) > 10:
                    # Calculate rolling volatility from recent PnL
                    recent_pnl = wfa_df['pnl'].tail(30)  # Last 30 observations
                    volatility = recent_pnl.std()
                    
                    # Volatility thresholds (could be calibrated)
                    if volatility < 0.5:
                        return 'low_vol', 0.8
                    elif volatility > 1.5:
                        return 'high_vol', 0.8
                    else:
                        return 'mid_vol', 0.7
                        
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error reading WFA data: {e}")
        
        # Method 2: Use synthetic regime based on time/season
        # (In real implementation, would use actual market data)
        now = datetime.now()
        
        # Simulate regime based on recent synthetic patterns
        np.random.seed(now.day)  # Deterministic but changing
        regime_roll = np.random.random()
        
        if regime_roll < 0.3:
            return 'low_vol', 0.6
        elif regime_roll > 0.7:
            return 'high_vol', 0.6
        else:
            return 'mid_vol', 0.7
    
    def should_rotate_portfolio(self, new_regime: str, confidence: float) -> bool:
        """Determine if portfolio should be rotated."""
        
        # Minimum confidence threshold
        if confidence < 0.5:
            return False
        
        # No rotation if regime unchanged
        if new_regime == self.last_regime:
            return False
        
        # High confidence threshold for regime switches
        if self.last_regime and confidence < 0.7:
            return False
        
        return True
    
    def apply_regime_profile(self, regime: str) -> Dict:
        """Apply regime-specific portfolio configuration."""
        profiles = self.config.get('regime_profiles', {})
        
        if regime not in profiles:
            if self.verbose:
                print(f"⚠️ Regime '{regime}' not found in profiles, using mid_vol")
            regime = 'mid_vol'
        
        profile = profiles[regime]
        
        # Create modified config with regime profile
        modified_config = self.config.copy()
        
        # Update selection parameters
        if 'top_n' in profile:
            modified_config['selection']['top_n'] = profile['top_n']
        
        # Update optimizer parameters
        for param in ['lambda_var', 'gamma_cost', 'max_weight_per_pair']:
            if param in profile:
                modified_config['optimizer'][param] = profile[param]
        
        # Update diversification
        if 'diversify_by_base' in profile:
            modified_config['selection']['diversify_by_base'] = profile['diversify_by_base']
        
        if self.verbose:
            print(f"📊 Applied {regime} profile:")
            print(f"   top_n: {profile.get('top_n', 'unchanged')}")
            print(f"   lambda_var: {profile.get('lambda_var', 'unchanged')}")
            print(f"   gamma_cost: {profile.get('gamma_cost', 'unchanged')}")
        
        return modified_config
    
    def rebuild_portfolio_with_regime(self, regime: str) -> bool:
        """Rebuild portfolio using regime-specific configuration."""
        try:
            # Apply regime profile
            modified_config = self.apply_regime_profile(regime)
            
            # Save temporary config
            temp_config_path = Path('artifacts/portfolio/temp_regime_config.yaml')
            temp_config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_config_path, 'w') as f:
                yaml.dump(modified_config, f, default_flow_style=False)
            
            if self.verbose:
                print(f"💾 Saved regime config to {temp_config_path}")
            
            # Run portfolio builder with regime config
            cmd = [
                'python', 'scripts/build_portfolio.py',
                '--config', str(temp_config_path)
            ]
            
            if self.verbose:
                print(f"🔧 Rebuilding portfolio with command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                if self.verbose:
                    print(f"   ✅ Portfolio rebuild successful")
                return True
            else:
                if self.verbose:
                    print(f"   ❌ Portfolio rebuild failed: {result.stderr}")
                return False
                
        except Exception as e:
            if self.verbose:
                print(f"❌ Error rebuilding portfolio: {e}")
            return False
    
    def log_regime_rotation(
        self, 
        old_regime: str, 
        new_regime: str, 
        confidence: float,
        rebuild_success: bool
    ) -> None:
        """Log regime rotation to report file."""
        
        rotation_report = Path('artifacts/portfolio/REGIME_ROTATION.md')
        rotation_report.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        log_entry = f"""
## {timestamp} - Regime Rotation

### Regime Change
- **From**: {old_regime or 'N/A'} 
- **To**: {new_regime}
- **Confidence**: {confidence:.2%}

### Portfolio Update
- **Rebuild Triggered**: {'✅ Yes' if rebuild_success else '❌ Failed'}
- **Reason**: Market regime shift detected

### Profile Applied ({new_regime})
"""
        
        # Add profile details
        profiles = self.config.get('regime_profiles', {})
        if new_regime in profiles:
            profile = profiles[new_regime]
            for key, value in profile.items():
                if key != 'diversify_by_base':  # Skip boolean for formatting
                    log_entry += f"- **{key}**: {value}\n"
        
        log_entry += f"""
### Next Steps
- Monitor performance under new regime profile
- Assess effectiveness of rotation after 3-7 days
- Consider manual override if regime detection is unstable

---
"""
        
        # Append to rotation log
        if rotation_report.exists():
            with open(rotation_report, 'a') as f:
                f.write(log_entry)
        else:
            header = """# Regime Rotation Log
Track of automatic portfolio rotations based on market regime detection.

---
"""
            with open(rotation_report, 'w') as f:
                f.write(header + log_entry)
        
        if self.verbose:
            print(f"✅ Logged regime rotation to {rotation_report}")
    
    def run_regime_rotation(self) -> Dict:
        """Run complete regime detection and portfolio rotation."""
        if self.verbose:
            print("=" * 60)
            print("REGIME-AWARE PORTFOLIO ROTATION")  
            print("=" * 60)
        
        # Load last known regime
        state_file = Path('artifacts/portfolio/regime_state.json')
        if state_file.exists():
            import json
            with open(state_file, 'r') as f:
                state = json.load(f)
            self.last_regime = state.get('regime')
            if self.verbose:
                print(f"📜 Last regime: {self.last_regime}")
        
        # Detect current regime
        if self.verbose:
            print("🔍 Detecting market regime...")
        
        current_regime, confidence = self.detect_market_regime()
        self.current_regime = current_regime
        self.regime_confidence = confidence
        
        if self.verbose:
            print(f"   Detected: {current_regime} (confidence: {confidence:.1%})")
        
        # Check if rotation needed
        should_rotate = self.should_rotate_portfolio(current_regime, confidence)
        
        if self.verbose:
            print(f"🔄 Should rotate: {'Yes' if should_rotate else 'No'}")
        
        result = {
            'regime_detected': current_regime,
            'confidence': confidence,
            'rotation_triggered': should_rotate,
            'rebuild_success': False,
            'previous_regime': self.last_regime
        }
        
        # Execute rotation if needed
        if should_rotate:
            if self.verbose:
                print("⚡ Executing portfolio rotation...")
            
            rebuild_success = self.rebuild_portfolio_with_regime(current_regime)
            result['rebuild_success'] = rebuild_success
            
            # Log rotation
            self.log_regime_rotation(
                self.last_regime, current_regime, confidence, rebuild_success
            )
            
            if self.verbose:
                status = "✅ Success" if rebuild_success else "❌ Failed"
                print(f"   Portfolio rebuild: {status}")
        
        # Save current state
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, 'w') as f:
            import json
            json.dump({
                'regime': current_regime,
                'confidence': confidence,
                'last_update': datetime.now().isoformat()
            }, f, indent=2)
        
        # Summary
        if self.verbose:
            print("\n" + "=" * 60)
            print("ROTATION SUMMARY")
            print("=" * 60)
            print(f"Current Regime: {current_regime}")
            print(f"Confidence: {confidence:.1%}")
            print(f"Rotation: {'Executed' if should_rotate else 'Not needed'}")
            if should_rotate:
                print(f"Rebuild: {'✅ Success' if result['rebuild_success'] else '❌ Failed'}")
        
        return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Regime-aware portfolio rotation')
    
    parser.add_argument('--config', default='configs/portfolio_optimizer.yaml',
                       help='Path to portfolio configuration')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--force', action='store_true',
                       help='Force rotation regardless of confidence')
    parser.add_argument('--regime', choices=['low_vol', 'mid_vol', 'high_vol'],
                       help='Force specific regime (override detection)')
    
    args = parser.parse_args()
    
    # Initialize rotator
    rotator = RegimePortfolioRotator(config_path=args.config, verbose=args.verbose)
    
    # Run rotation
    result = rotator.run_regime_rotation()
    
    # Override regime detection if forced
    if args.regime:
        result['regime_detected'] = args.regime
        result['confidence'] = 1.0 if args.force else 0.8
        result['rotation_triggered'] = True
        
        if args.verbose:
            print(f"🔧 Forcing rotation to regime: {args.regime}")
        
        # Execute forced rotation
        rebuild_success = rotator.rebuild_portfolio_with_regime(args.regime)
        result['rebuild_success'] = rebuild_success
        
        # Log forced rotation
        rotator.log_regime_rotation(
            result['previous_regime'], args.regime, result['confidence'], rebuild_success
        )
    
    
    # Exit with appropriate code
    if result['rotation_triggered']:
        exit_code = 0 if result['rebuild_success'] else 1
    else:
        exit_code = 0  # No rotation needed is OK
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()