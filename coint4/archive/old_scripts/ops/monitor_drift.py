#!/usr/bin/env python3
"""
Drift monitoring with automatic reactions.
Tracks performance degradation and triggers risk reduction or portfolio rebuild.
"""

import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class DriftMonitor:
    """Monitor performance drift and trigger automatic responses."""
    
    def __init__(self, config_path: str, verbose: bool = False):
        """Initialize drift monitor."""
        self.verbose = verbose
        self.config_path = config_path
        self.config = self._load_config()
        
        self.drift_status = "OK"  # OK | WARN | FAIL
        self.degradation_level = 0  # 0, 1, 2, 3
        self.actions_taken = []
        
        # Create output directories
        for output_path in self.config['outputs'].values():
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load drift monitoring configuration."""
        config_path = Path(self.config_path)
        
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if self.verbose:
                    print(f"✅ Loaded config from {config_path}")
                return config
        else:
            raise FileNotFoundError(f"Config not found: {config_path}")
    
    def load_confidence_data(self) -> pd.DataFrame:
        """Load confidence intervals from uncertainty analysis."""
        confidence_file = self.config['data_sources']['confidence_file']
        confidence_path = Path(confidence_file)
        
        if not confidence_path.exists():
            if self.verbose:
                print(f"⚠️ Confidence file not found: {confidence_file}")
            # Return dummy data
            return pd.DataFrame({
                'pair': ['PORTFOLIO'],
                'metric': ['sharpe'],
                'p05': [0.3],
                'p50': [0.6],
                'p95': [1.0],
                'observed': [0.7]
            })
        
        confidence_df = pd.read_csv(confidence_path)
        if self.verbose:
            print(f"✅ Loaded confidence data: {len(confidence_df)} entries")
        
        return confidence_df
    
    def load_recent_performance(self) -> pd.DataFrame:
        """Load recent performance data for drift analysis."""
        wfa_file = self.config['data_sources']['wfa_results']
        wfa_path = Path(wfa_file)
        
        if not wfa_path.exists():
            if self.verbose:
                print(f"⚠️ WFA results not found: {wfa_file}")
            
            # Generate synthetic recent performance
            dates = pd.date_range(datetime.now() - timedelta(days=90), periods=90, freq='D')
            
            # Simulate degrading performance
            sharpe_trend = 0.8 + 0.3 * np.exp(-np.arange(90) / 30.0)  # Decay
            psr_trend = 0.85 + 0.15 * np.exp(-np.arange(90) / 45.0)
            
            synthetic_data = pd.DataFrame({
                'date': dates,
                'sharpe': sharpe_trend + np.random.normal(0, 0.1, 90),
                'psr': psr_trend + np.random.normal(0, 0.05, 90),
                'trades': np.random.poisson(25, 90),
                'pnl': np.random.normal(0.02, 0.5, 90)
            })
            
            return synthetic_data
        
        # Load actual WFA data
        wfa_df = pd.read_csv(wfa_path)
        
        # Convert to time series if needed
        if 'timestamp' in wfa_df.columns:
            wfa_df['date'] = pd.to_datetime(wfa_df['timestamp'])
        elif 'date' not in wfa_df.columns:
            # Add synthetic dates
            wfa_df['date'] = pd.date_range(
                end=datetime.now(), periods=len(wfa_df), freq='D'
            )
        
        return wfa_df
    
    def calculate_drift_metrics(self, performance_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate drift metrics comparing short vs long windows."""
        windows = self.config['windows']
        short_days = windows['short_days']
        long_days = windows['long_days']
        
        # Sort by date
        perf_df = performance_data.sort_values('date')
        
        # Define windows
        cutoff_short = datetime.now() - timedelta(days=short_days)
        cutoff_long = datetime.now() - timedelta(days=long_days)
        
        short_window = perf_df[perf_df['date'] >= cutoff_short]
        long_window = perf_df[perf_df['date'] >= cutoff_long]
        
        drift_metrics = {}
        
        # Sharpe drift
        if 'sharpe' in perf_df.columns:
            short_sharpe = short_window['sharpe'].mean()
            long_sharpe = long_window['sharpe'].mean()
            
            if long_sharpe != 0:
                sharpe_change = (short_sharpe - long_sharpe) / abs(long_sharpe)
                drift_metrics['sharpe_short'] = short_sharpe
                drift_metrics['sharpe_long'] = long_sharpe
                drift_metrics['sharpe_change'] = sharpe_change
                drift_metrics['sharpe_drop'] = -sharpe_change if sharpe_change < 0 else 0
            else:
                drift_metrics['sharpe_drop'] = 0
        
        # PSR drift
        if 'psr' in perf_df.columns:
            short_psr = short_window['psr'].mean()
            long_psr = long_window['psr'].mean()
            
            psr_change = short_psr - long_psr
            drift_metrics['psr_short'] = short_psr
            drift_metrics['psr_long'] = long_psr
            drift_metrics['psr_change'] = psr_change
            drift_metrics['psr_drop'] = -psr_change if psr_change < 0 else 0
        
        # Data quality
        drift_metrics['short_obs'] = len(short_window)
        drift_metrics['long_obs'] = len(long_window)
        
        return drift_metrics
    
    def assess_drift_status(
        self, 
        confidence_data: pd.DataFrame,
        drift_metrics: Dict[str, float]
    ) -> Tuple[str, int]:
        """Assess overall drift status and degradation level.
        
        Returns:
            Tuple of (status, level) where:
            - status: OK | WARN | FAIL
            - level: 0-3 (degradation severity)
        """
        thresholds = self.config['thresholds']
        
        # Get portfolio confidence bounds
        portfolio_confidence = confidence_data[
            confidence_data['pair'] == 'PORTFOLIO'
        ]
        
        # Check P5 bounds
        sharpe_p05 = 0
        psr_p05 = 0.5
        
        if not portfolio_confidence.empty:
            sharpe_row = portfolio_confidence[portfolio_confidence['metric'] == 'sharpe']
            if not sharpe_row.empty:
                sharpe_p05 = sharpe_row.iloc[0]['p05']
            
            psr_row = portfolio_confidence[portfolio_confidence['metric'] == 'psr']
            if not psr_row.empty:
                psr_p05 = psr_row.iloc[0]['p05']
        
        # Check performance drops
        sharpe_drop = drift_metrics.get('sharpe_drop', 0)
        psr_drop = drift_metrics.get('psr_drop', 0)
        
        # Determine status and level
        status = "OK"
        level = 0
        
        # Level 3: Severe degradation
        if (sharpe_p05 < thresholds.get('level_3_psr_min', 0.50) or
            sharpe_drop > thresholds.get('level_3_sharpe_drop', 0.60) or
            psr_p05 < thresholds.get('level_3_psr_min', 0.50)):
            status = "FAIL"
            level = 3
            
        # Level 2: Moderate degradation
        elif (sharpe_p05 < thresholds.get('level_2_psr_min', 0.70) or
              sharpe_drop > thresholds.get('level_2_sharpe_drop', 0.40) or
              psr_p05 < thresholds.get('level_2_psr_min', 0.70)):
            status = "FAIL" 
            level = 2
            
        # Level 1: Mild degradation
        elif (sharpe_p05 < thresholds.get('level_1_psr_min', 0.85) or
              sharpe_drop > thresholds.get('level_1_sharpe_drop', 0.25) or
              psr_p05 < thresholds.get('level_1_psr_min', 0.85)):
            status = "WARN"
            level = 1
        
        # Warning conditions
        elif (sharpe_drop > thresholds.get('sharpe_drop_tol', 0.35) or
              psr_drop > thresholds.get('psr_drop_tol', 0.20)):
            status = "WARN"
            level = 0
        
        return status, level
    
    def execute_actions(self, status: str, level: int) -> List[str]:
        """Execute automatic response actions based on drift status."""
        actions_config = self.config['actions']
        actions_taken = []
        
        if status == "OK":
            return actions_taken
        
        # Risk reduction (derisk)
        if level > 0:
            scale_levels = actions_config.get('derisk_scale', [0.75, 0.5, 0.25])
            
            if level <= len(scale_levels):
                scale_factor = scale_levels[level - 1]
                
                # Execute derisk action
                success = self._execute_derisk(scale_factor)
                if success:
                    actions_taken.append(f"Derisk: Scaled positions to {scale_factor:.0%}")
                else:
                    actions_taken.append(f"Derisk: FAILED to scale positions to {scale_factor:.0%}")
        
        # Portfolio rebuild for severe drift
        if level >= 2 and actions_config.get('rebuild_portfolio', False):
            success = self._execute_rebuild_portfolio()
            if success:
                actions_taken.append("Portfolio: Triggered rebuild due to severe drift")
            else:
                actions_taken.append("Portfolio: FAILED to trigger rebuild")
        
        return actions_taken
    
    def _execute_derisk(self, scale_factor: float) -> bool:
        """Execute position scaling (derisk) action."""
        try:
            # Update paper week with derisk scale
            cmd = [
                'python', 'scripts/run_paper_week.py',
                '--derisk-scale', str(scale_factor),
                '--quiet'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                if self.verbose:
                    print(f"   ✅ Derisked positions by {scale_factor:.0%}")
                return True
            else:
                if self.verbose:
                    print(f"   ❌ Derisk failed: {result.stderr}")
                return False
                
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Derisk error: {e}")
            return False
    
    def _execute_rebuild_portfolio(self) -> bool:
        """Execute portfolio rebuild action."""
        try:
            cmd = [
                'python', 'scripts/build_portfolio.py',
                '--config', 'configs/portfolio_optimizer.yaml',
                '--reason', 'drift_degradation'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                if self.verbose:
                    print(f"   ✅ Portfolio rebuild triggered")
                return True
            else:
                if self.verbose:
                    print(f"   ❌ Portfolio rebuild failed: {result.stderr}")
                return False
                
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Portfolio rebuild error: {e}")
            return False
    
    def generate_dashboard(
        self,
        confidence_data: pd.DataFrame,
        drift_metrics: Dict[str, float],
        status: str,
        level: int,
        actions: List[str]
    ) -> str:
        """Generate drift monitoring dashboard."""
        dashboard_content = f"""# Drift Monitoring Dashboard
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Status Overview
- **Overall Status**: {self._status_emoji(status)} **{status}**
- **Degradation Level**: {level}/3
- **Actions Taken**: {len(actions)} action(s)

## Performance Drift Analysis

### Recent vs Baseline Comparison
| Metric | Short Window | Long Window | Change | Drop |
|--------|--------------|-------------|--------|------|
| Sharpe | {drift_metrics.get('sharpe_short', 0):.3f} | {drift_metrics.get('sharpe_long', 0):.3f} | {drift_metrics.get('sharpe_change', 0):.1%} | {drift_metrics.get('sharpe_drop', 0):.1%} |
| PSR | {drift_metrics.get('psr_short', 0):.3f} | {drift_metrics.get('psr_long', 0):.3f} | {drift_metrics.get('psr_change', 0):.3f} | {drift_metrics.get('psr_drop', 0):.3f} |

### Confidence Bounds Analysis
"""
        
        # Add confidence analysis
        portfolio_confidence = confidence_data[confidence_data['pair'] == 'PORTFOLIO']
        
        if not portfolio_confidence.empty:
            dashboard_content += "\n| Metric | P05 | P50 | P95 | Observed | Status |\n|--------|-----|-----|-----|----------|--------|\n"
            
            for _, row in portfolio_confidence.iterrows():
                metric = row['metric']
                p05 = row['p05']
                p95 = row['p95']
                observed = row['observed']
                
                # Status based on thresholds
                if metric == 'sharpe':
                    threshold = self.config['thresholds'].get('sharpe_p5_min', 0.60)
                    status_icon = "✅" if p05 >= threshold else "🚨"
                elif metric == 'psr':
                    threshold = self.config['thresholds'].get('psr_p5_min', 0.90)
                    status_icon = "✅" if p05 >= threshold else "🚨"
                else:
                    status_icon = "➖"
                
                dashboard_content += f"| {metric.upper()} | {p05:.3f} | {row['p50']:.3f} | {p95:.3f} | {observed:.3f} | {status_icon} |\n"
        
        # Add actions section
        if actions:
            dashboard_content += f"\n## Actions Taken\n"
            for i, action in enumerate(actions, 1):
                dashboard_content += f"{i}. {action}\n"
        
        # Add thresholds reference
        thresholds = self.config['thresholds']
        dashboard_content += f"""
## Threshold Reference
- **Sharpe P05 Min**: {thresholds.get('sharpe_p5_min', 0.60)}
- **PSR P05 Min**: {thresholds.get('psr_p5_min', 0.90)}
- **Max Sharpe Drop**: {thresholds.get('sharpe_drop_tol', 0.35):.0%}
- **Max PSR Drop**: {thresholds.get('psr_drop_tol', 0.20):.0%}

## Data Quality
- Short window observations: {drift_metrics.get('short_obs', 0)}
- Long window observations: {drift_metrics.get('long_obs', 0)}
- Window sizes: {self.config['windows']['short_days']}d / {self.config['windows']['long_days']}d

## Next Steps
"""
        
        if status == "FAIL":
            dashboard_content += "- 🚨 **IMMEDIATE ACTION REQUIRED**: Review degradation causes\n"
            dashboard_content += "- 🔧 **Consider**: Manual portfolio review and parameter adjustment\n"
            if level >= 2:
                dashboard_content += "- 📊 **Monitor**: Portfolio rebuild in progress\n"
        elif status == "WARN":
            dashboard_content += "- ⚠️ **Monitor closely**: Watch for further degradation\n"
            dashboard_content += "- 📈 **Consider**: Risk reduction if trend continues\n"
        else:
            dashboard_content += "- ✅ **Continue monitoring**: Performance within acceptable bounds\n"
        
        return dashboard_content
    
    def _status_emoji(self, status: str) -> str:
        """Get status emoji."""
        return {"OK": "✅", "WARN": "⚠️", "FAIL": "🚨"}.get(status, "❓")
    
    def log_actions(self, actions: List[str], status: str, level: int) -> None:
        """Log actions taken to ACTIONS_TAKEN.md."""
        actions_file = self.config['outputs']['actions_log']
        actions_path = Path(actions_file)
        
        # Create or append to actions log
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        log_entry = f"""
## {timestamp} - Status: {status} (Level {level})

"""
        
        if actions:
            log_entry += "### Actions Executed:\n"
            for action in actions:
                log_entry += f"- {action}\n"
        else:
            log_entry += "### Actions: None required\n"
        
        log_entry += "\n---\n"
        
        # Append to file
        if actions_path.exists():
            with open(actions_path, 'a') as f:
                f.write(log_entry)
        else:
            header = """# Actions Taken Log
Record of automatic responses to performance drift.

---
"""
            with open(actions_path, 'w') as f:
                f.write(header + log_entry)
        
        if self.verbose:
            print(f"✅ Logged {len(actions)} actions to {actions_file}")
    
    def run_monitoring(self) -> Dict[str, Any]:
        """Run complete drift monitoring cycle."""
        if self.verbose:
            print("=" * 60)
            print("DRIFT MONITORING")
            print("=" * 60)
        
        # Load data
        if self.verbose:
            print("\n📊 Loading confidence and performance data...")
        
        confidence_data = self.load_confidence_data()
        performance_data = self.load_recent_performance()
        
        # Calculate drift metrics
        if self.verbose:
            print("📈 Calculating drift metrics...")
        
        drift_metrics = self.calculate_drift_metrics(performance_data)
        
        # Assess status
        if self.verbose:
            print("🔍 Assessing drift status...")
        
        status, level = self.assess_drift_status(confidence_data, drift_metrics)
        
        if self.verbose:
            print(f"   Status: {self._status_emoji(status)} {status} (Level {level})")
        
        # Execute actions
        if self.verbose and (status != "OK"):
            print("⚡ Executing automatic responses...")
        
        actions = self.execute_actions(status, level)
        
        if self.verbose:
            print(f"   Actions taken: {len(actions)}")
        
        # Generate reports
        if self.verbose:
            print("📝 Generating reports...")
        
        dashboard_content = self.generate_dashboard(
            confidence_data, drift_metrics, status, level, actions
        )
        
        # Save dashboard
        dashboard_file = self.config['outputs']['dashboard']
        with open(dashboard_file, 'w') as f:
            f.write(dashboard_content)
        
        # Log actions
        if actions or status != "OK":
            self.log_actions(actions, status, level)
        
        # Save drift data
        drift_data_file = self.config['outputs']['drift_data']
        drift_record = {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'level': level,
            'actions_count': len(actions),
            **drift_metrics
        }
        
        # Append to CSV
        drift_df = pd.DataFrame([drift_record])
        if Path(drift_data_file).exists():
            existing_df = pd.read_csv(drift_data_file)
            drift_df = pd.concat([existing_df, drift_df], ignore_index=True)
        
        drift_df.to_csv(drift_data_file, index=False)
        
        # Summary
        if self.verbose:
            print("\n" + "=" * 60)
            print("MONITORING SUMMARY")
            print("=" * 60)
            print(f"Status: {self._status_emoji(status)} {status}")
            print(f"Level: {level}/3")
            print(f"Actions: {len(actions)}")
            if actions:
                for action in actions:
                    print(f"  - {action}")
            print(f"\nReports: {dashboard_file}")
        
        return {
            'status': status,
            'level': level,
            'actions': actions,
            'metrics': drift_metrics,
            'reports': {
                'dashboard': dashboard_file,
                'actions_log': self.config['outputs']['actions_log']
            }
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Monitor performance drift and trigger responses')
    
    parser.add_argument('--config', default='configs/drift_monitor.yaml',
                       help='Path to drift monitoring configuration')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - no actions executed')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = DriftMonitor(config_path=args.config, verbose=args.verbose)
    
    # Override actions for dry run
    if args.dry_run:
        monitor.config['actions']['derisk_scale'] = []
        monitor.config['actions']['rebuild_portfolio'] = False
        if args.verbose:
            print("🔒 DRY RUN: No actions will be executed")
    
    # Run monitoring
    result = monitor.run_monitoring()
    
    # Exit with appropriate code
    exit_code = 0 if result['status'] in ['OK', 'WARN'] else 1
    sys.exit(exit_code)


if __name__ == '__main__':
    main()