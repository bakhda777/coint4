#!/usr/bin/env python3
"""
Local task scheduler for uncertainty and drift monitoring.
Runs daily/weekly monitoring tasks without external dependencies.
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json


class LocalScheduler:
    """Local task scheduler for monitoring and maintenance."""
    
    def __init__(self, verbose: bool = False, dry_run: bool = False):
        """Initialize scheduler."""
        self.verbose = verbose
        self.dry_run = dry_run
        self.results = []
        
        # Create scheduler state directory
        self.state_dir = Path('artifacts/scheduler')
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / 'scheduler_state.json'
        
        # Load scheduler state
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load scheduler state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Default state
        return {
            'last_daily': None,
            'last_weekly': None,
            'consecutive_failures': 0,
            'maintenance_mode': False
        }
    
    def _save_state(self) -> None:
        """Save scheduler state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def _run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and return success status."""
        if self.verbose:
            print(f"   🔧 Running: {description}")
            print(f"      Command: {' '.join(cmd)}")
        
        if self.dry_run:
            print(f"   🔒 DRY RUN: Would execute {description}")
            return True
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=Path.cwd(),
                timeout=300  # 5 minute timeout
            )
            
            success = result.returncode == 0
            
            # Log result
            task_result = {
                'task': description,
                'timestamp': datetime.now().isoformat(),
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout[:500] if result.stdout else '',  # Limit output
                'stderr': result.stderr[:500] if result.stderr else ''
            }
            
            self.results.append(task_result)
            
            if self.verbose:
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"      Result: {status}")
                if not success and result.stderr:
                    print(f"      Error: {result.stderr[:100]}")
            
            return success
            
        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"      Result: ⏰ TIMEOUT")
            return False
        except Exception as e:
            if self.verbose:
                print(f"      Result: ❌ ERROR ({e})")
            return False
    
    def run_daily_tasks(self) -> bool:
        """Run daily monitoring tasks."""
        if self.verbose:
            print("\n📅 DAILY TASKS")
            print("=" * 40)
        
        tasks_success = []
        
        # 1. Uncertainty analysis (light mode)
        if self.verbose:
            print("\n🔍 Running uncertainty analysis...")
        
        success = self._run_command([
            'python', 'scripts/run_uncertainty.py', 
            '--quick',
            '--output-dir', 'artifacts/uncertainty'
        ], 'Daily uncertainty analysis')
        tasks_success.append(success)
        
        # 2. Drift monitoring  
        if self.verbose:
            print("\n📉 Running drift monitoring...")
        
        success = self._run_command([
            'python', 'scripts/monitor_drift.py',
            '--config', 'configs/drift_monitor.yaml',
            '--verbose' if self.verbose else ''
        ], 'Drift monitoring and response')
        tasks_success.append(success)
        
        # 3. Paper week with potential derisk scaling
        # Check if drift monitoring triggered derisk
        drift_state_file = Path('artifacts/monitoring/drift_data.csv')
        derisk_needed = False
        derisk_scale = 1.0
        
        if drift_state_file.exists():
            try:
                import pandas as pd
                drift_df = pd.read_csv(drift_state_file)
                if not drift_df.empty:
                    latest_entry = drift_df.iloc[-1]
                    if latest_entry.get('status') == 'FAIL' and latest_entry.get('level', 0) > 0:
                        derisk_needed = True
                        # Scale based on degradation level
                        level = latest_entry.get('level', 1)
                        derisk_scale = [1.0, 0.75, 0.5, 0.25][min(level, 3)]
            except Exception:
                pass
        
        if self.verbose:
            print("\n📊 Running paper week simulation...")
            if derisk_needed:
                print(f"   ⚠️ Applying derisk scaling: {derisk_scale:.0%}")
        
        paper_week_cmd = [
            'python', 'scripts/run_paper_week.py',
            '--pairs-file', 'bench/pairs_portfolio.yaml',
            '--portfolio-weights', 'artifacts/portfolio/weights.csv'
        ]
        
        if derisk_needed:
            paper_week_cmd.extend(['--derisk-scale', str(derisk_scale)])
        
        success = self._run_command(paper_week_cmd, 'Paper week simulation')
        tasks_success.append(success)
        
        # Update state
        self.state['last_daily'] = datetime.now().isoformat()
        
        # Track consecutive failures
        if all(tasks_success):
            self.state['consecutive_failures'] = 0
        else:
            self.state['consecutive_failures'] += 1
        
        overall_success = all(tasks_success)
        
        if self.verbose:
            print(f"\n📊 Daily tasks completed: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
            print(f"   Tasks passed: {sum(tasks_success)}/{len(tasks_success)}")
        
        return overall_success
    
    def run_weekly_tasks(self) -> bool:
        """Run weekly maintenance tasks."""
        if self.verbose:
            print("\n📅 WEEKLY TASKS")
            print("=" * 40)
        
        tasks_success = []
        
        # 1. Portfolio rebuild (if needed)
        regime_state_file = Path('artifacts/portfolio/regime_state.json')
        regime_changed = False
        
        if regime_state_file.exists():
            try:
                with open(regime_state_file, 'r') as f:
                    regime_state = json.load(f)
                
                last_update = datetime.fromisoformat(regime_state.get('last_update', '2000-01-01'))
                
                # Rebuild if regime hasn't been updated in over a week
                if datetime.now() - last_update > timedelta(days=7):
                    regime_changed = True
            except Exception:
                regime_changed = True  # Rebuild if can't read state
        
        if self.verbose:
            print("\n🔄 Checking portfolio regime...")
        
        # Run regime rotation (will detect current regime and rotate if needed)
        success = self._run_command([
            'python', 'scripts/rotate_portfolio_by_regime.py',
            '--config', 'configs/portfolio_optimizer.yaml',
            '--verbose' if self.verbose else ''
        ], 'Regime-based portfolio rotation')
        tasks_success.append(success)
        
        # 2. Full WFA if major changes occurred
        major_changes = (
            self.state.get('consecutive_failures', 0) >= 2 or
            regime_changed or
            not Path('artifacts/wfa/WFA_REPORT.md').exists()
        )
        
        if major_changes:
            if self.verbose:
                print("\n🔬 Running full WFA analysis...")
            
            success = self._run_command([
                'python', 'scripts/run_walk_forward.py',
                '--pairs-file', 'bench/pairs_portfolio.yaml',
                '--portfolio-weights', 'artifacts/portfolio/weights.csv'
            ], 'Full walk-forward analysis')
            tasks_success.append(success)
        
        # 3. Comprehensive uncertainty analysis
        if self.verbose:
            print("\n📊 Running comprehensive uncertainty analysis...")
        
        success = self._run_command([
            'python', 'scripts/run_uncertainty.py',
            '--n-bootstrap', '1000',
            '--output-dir', 'artifacts/uncertainty'
        ], 'Comprehensive uncertainty analysis')
        tasks_success.append(success)
        
        # 4. CI gates validation
        if self.verbose:
            print("\n🚪 Running CI gates validation...")
        
        success = self._run_command([
            'python', 'scripts/ci_gates.py',
            '--config', 'configs/ci_gates.yaml',
            '--verbose' if self.verbose else ''
        ], 'CI gates validation')
        tasks_success.append(success)
        
        # Update state
        self.state['last_weekly'] = datetime.now().isoformat()
        
        overall_success = all(tasks_success)
        
        if self.verbose:
            print(f"\n📊 Weekly tasks completed: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
            print(f"   Tasks passed: {sum(tasks_success)}/{len(tasks_success)}")
        
        return overall_success
    
    def should_run_daily(self) -> bool:
        """Check if daily tasks should be run."""
        last_daily = self.state.get('last_daily')
        
        if not last_daily:
            return True
        
        try:
            last_run = datetime.fromisoformat(last_daily)
            return datetime.now() - last_run > timedelta(hours=20)  # Allow some flexibility
        except Exception:
            return True
    
    def should_run_weekly(self) -> bool:
        """Check if weekly tasks should be run."""
        last_weekly = self.state.get('last_weekly')
        
        if not last_weekly:
            return True
        
        try:
            last_run = datetime.fromisoformat(last_weekly)
            return datetime.now() - last_run > timedelta(days=6)  # Allow some flexibility
        except Exception:
            return True
    
    def generate_report(self) -> str:
        """Generate scheduler execution report."""
        report = f"""# Scheduler Execution Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Execution Summary
- Total tasks: {len(self.results)}
- Successful: {sum(1 for r in self.results if r['success'])}
- Failed: {sum(1 for r in self.results if not r['success'])}

## Task Results
"""
        
        for result in self.results:
            status = "✅" if result['success'] else "❌"
            report += f"- {status} {result['task']} ({result['timestamp']})\n"
            
            if not result['success']:
                if result['stderr']:
                    report += f"  Error: {result['stderr'][:200]}\n"
        
        report += f"""
## State
- Last daily: {self.state.get('last_daily', 'Never')}
- Last weekly: {self.state.get('last_weekly', 'Never')}
- Consecutive failures: {self.state.get('consecutive_failures', 0)}

## Next Scheduled Runs
- Daily: {'Due now' if self.should_run_daily() else 'Not due'}
- Weekly: {'Due now' if self.should_run_weekly() else 'Not due'}
"""
        
        return report
    
    def run_scheduler(self, force_daily: bool = False, force_weekly: bool = False) -> bool:
        """Run scheduler with appropriate tasks."""
        if self.verbose:
            print("🕐 LOCAL SCHEDULER")
            print("=" * 60)
        
        overall_success = True
        
        # Determine what to run
        run_daily = force_daily or self.should_run_daily()
        run_weekly = force_weekly or self.should_run_weekly()
        
        if not run_daily and not run_weekly:
            if self.verbose:
                print("ℹ️ No tasks scheduled to run")
            return True
        
        if self.verbose:
            print(f"📋 Scheduled tasks:")
            if run_daily:
                print(f"   - Daily monitoring tasks")
            if run_weekly:
                print(f"   - Weekly maintenance tasks")
        
        # Run daily tasks
        if run_daily:
            daily_success = self.run_daily_tasks()
            overall_success &= daily_success
        
        # Run weekly tasks
        if run_weekly:
            weekly_success = self.run_weekly_tasks()
            overall_success &= weekly_success
        
        # Save state
        self._save_state()
        
        # Generate report
        report_content = self.generate_report()
        report_file = self.state_dir / f'SCHEDULER_REPORT_{datetime.now().strftime("%Y%m%d_%H%M")}.md'
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        # Summary
        if self.verbose:
            print("\n" + "=" * 60)
            print("SCHEDULER SUMMARY")
            print("=" * 60)
            print(f"Overall result: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
            print(f"Tasks executed: {len(self.results)}")
            print(f"Report: {report_file}")
        
        return overall_success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Local scheduler for uncertainty and drift monitoring')
    
    parser.add_argument('--daily', action='store_true',
                       help='Force run daily tasks')
    parser.add_argument('--weekly', action='store_true', 
                       help='Force run weekly tasks')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - show what would be executed')
    
    args = parser.parse_args()
    
    # Initialize scheduler
    scheduler = LocalScheduler(verbose=args.verbose, dry_run=args.dry_run)
    
    if args.dry_run and args.verbose:
        print("🔒 DRY RUN MODE: No commands will be executed")
    
    # Run scheduler
    success = scheduler.run_scheduler(
        force_daily=args.daily,
        force_weekly=args.weekly
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()