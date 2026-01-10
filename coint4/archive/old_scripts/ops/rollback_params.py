#!/usr/bin/env python3
"""
Rollback promoted parameters to previous versions.
"""

import sys
from pathlib import Path
from datetime import datetime
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class ParameterRollback:
    """Manages parameter rollback."""
    
    def __init__(self):
        """Initialize rollback system."""
        self.locked_dir = Path('configs/locked')
        self.current_dir = self.locked_dir / 'current'
        self.catalog_path = Path('artifacts/governance/PARAMS_CATALOG.md')
    
    def list_versions(self, pair: str = 'BTCETH', timeframe: str = 'H1') -> list:
        """List available parameter versions."""
        pattern = f"params_{pair}_{timeframe}_*.yaml"
        versions = sorted(self.locked_dir.glob(pattern))
        return versions
    
    def rollback(self, pair: str = 'BTCETH', timeframe: str = 'H1', 
                 week: str = None, index: int = -2) -> bool:
        """Rollback to specific version.
        
        Args:
            pair: Trading pair
            timeframe: Timeframe
            week: Specific week to rollback to (optional)
            index: Version index (-1 = current, -2 = previous)
        """
        versions = self.list_versions(pair, timeframe)
        
        if len(versions) < 2:
            print(f"❌ Not enough versions to rollback (found {len(versions)})")
            return False
        
        # Select target version
        if week:
            # Find specific week
            target = None
            for v in versions:
                if week in str(v):
                    target = v
                    break
            if not target:
                print(f"❌ Version for week {week} not found")
                return False
        else:
            # Use index
            try:
                target = versions[index]
            except IndexError:
                print(f"❌ Invalid version index {index}")
                return False
        
        # Update symlink
        current_link = self.current_dir / f"{pair}_{timeframe}.yaml"
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()
        
        # Create new symlink
        relative_path = Path('..') / target.name
        current_link.symlink_to(relative_path)
        
        # Log rollback
        self.log_rollback(pair, timeframe, target)
        
        print(f"✅ Rolled back to: {target.name}")
        print(f"   Current link updated: {current_link}")
        
        return True
    
    def log_rollback(self, pair: str, timeframe: str, target: Path):
        """Log rollback in catalog."""
        if not self.catalog_path.exists():
            return
        
        with open(self.catalog_path, 'a') as f:
            date = datetime.now().strftime('%Y-%m-%d %H:%M')
            f.write(f"| {date} | {pair} | {timeframe} | - | - | - | - | - | 🔄 Rolled back to {target.name} |\n")
    
    def show_current(self):
        """Show current parameter versions."""
        print("📊 Current Parameter Versions:")
        print("-" * 50)
        
        for link in self.current_dir.glob("*.yaml"):
            if link.is_symlink():
                target = link.readlink()
                print(f"  {link.name} -> {target}")
                
                # Read and show key params
                try:
                    with open(link) as f:
                        config = yaml.safe_load(f)
                        params = config.get('parameters', {})
                        meta = config.get('metadata', {})
                        
                        print(f"    Sharpe: {meta.get('validation', {}).get('sharpe', 'N/A')}")
                        print(f"    Z-threshold: {params.get('zscore_threshold', 'N/A')}")
                        print(f"    Window: {params.get('rolling_window', 'N/A')}")
                except:
                    pass


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Rollback promoted parameters')
    parser.add_argument('--pair', default='BTCETH', help='Trading pair')
    parser.add_argument('--timeframe', default='H1', help='Timeframe')
    parser.add_argument('--week', help='Rollback to specific week')
    parser.add_argument('--index', type=int, default=-2, 
                       help='Version index (-1=current, -2=previous)')
    parser.add_argument('--list', action='store_true', help='List available versions')
    parser.add_argument('--show', action='store_true', help='Show current versions')
    
    args = parser.parse_args()
    
    rollback = ParameterRollback()
    
    if args.show:
        rollback.show_current()
    elif args.list:
        versions = rollback.list_versions(args.pair, args.timeframe)
        print(f"📋 Available versions for {args.pair} {args.timeframe}:")
        for v in versions:
            print(f"  - {v.name}")
    else:
        success = rollback.rollback(
            pair=args.pair,
            timeframe=args.timeframe,
            week=args.week,
            index=args.index
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()