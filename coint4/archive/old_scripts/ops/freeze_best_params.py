#!/usr/bin/env python3
"""Freeze best parameters weekly for production deployment."""

import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import hashlib


def get_latest_optuna_results() -> dict:
    """Get latest Optuna optimization results."""
    optuna_dir = Path("artifacts/optuna")
    if not optuna_dir.exists():
        return None
    
    # Find latest results file
    result_files = list(optuna_dir.glob("*_results.json"))
    if not result_files:
        return None
    
    latest = max(result_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest, 'r') as f:
        return json.load(f)


def get_latest_wfa_params() -> dict:
    """Get latest WFA best parameters."""
    wfa_dir = Path("artifacts/wfa")
    if not wfa_dir.exists():
        return None
    
    # Check for params stability file
    stability_file = wfa_dir / "params_stability.csv"
    if stability_file.exists():
        import pandas as pd
        df = pd.read_csv(stability_file)
        
        # Get median values for stable params
        if 'verdict' in df.columns and 'stable' in df['verdict'].values:
            return {
                'zscore_threshold': df['zscore_threshold_median'].iloc[0],
                'zscore_exit': df['zscore_exit_median'].iloc[0],
                'rolling_window': int(df['rolling_window_median'].iloc[0]),
                'max_holding_days': int(df['max_holding_days_median'].iloc[0])
            }
    
    return None


def calculate_params_hash(params: dict) -> str:
    """Calculate hash of parameters for versioning."""
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()[:8]


def freeze_parameters(source: str = "auto", override_params: dict = None):
    """Freeze parameters for production deployment."""
    
    # Determine week number
    now = datetime.now()
    week_num = now.isocalendar()[1]
    year = now.year
    week_id = f"{year}_W{week_num:02d}"
    
    # Get parameters based on source
    if override_params:
        params = override_params
        source_name = "manual"
    elif source == "optuna":
        results = get_latest_optuna_results()
        if results:
            params = results['best_params']
            source_name = f"optuna_{results['study_name']}"
        else:
            print("No Optuna results found")
            return
    elif source == "wfa":
        params = get_latest_wfa_params()
        if params:
            source_name = "wfa_stable"
        else:
            print("No stable WFA parameters found")
            return
    else:  # auto
        # Try WFA first (more stable), then Optuna
        params = get_latest_wfa_params()
        if params:
            source_name = "wfa_stable"
        else:
            results = get_latest_optuna_results()
            if results:
                params = results['best_params']
                source_name = f"optuna_{results['study_name']}"
            else:
                print("No parameters found from any source")
                return
    
    # Create frozen params structure
    params_hash = calculate_params_hash(params)
    
    frozen = {
        'version': f"{week_id}_{params_hash}",
        'week_id': week_id,
        'frozen_at': now.isoformat(),
        'valid_from': now.strftime("%Y-%m-%d"),
        'valid_to': (now + timedelta(days=7)).strftime("%Y-%m-%d"),
        'source': source_name,
        'hash': params_hash,
        'parameters': params,
        'metadata': {
            'frozen_by': 'freeze_best_params.py',
            'environment': 'production',
            'approved': False,
            'deployed': False
        }
    }
    
    # Save to frozen params directory
    frozen_dir = Path("configs/frozen")
    frozen_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON version
    json_path = frozen_dir / f"params_{week_id}.json"
    with open(json_path, 'w') as f:
        json.dump(frozen, f, indent=2)
    
    print(f"Frozen parameters saved to {json_path}")
    
    # Save YAML version for config override
    yaml_path = frozen_dir / f"params_{week_id}.yaml"
    yaml_config = {
        'signals': {
            'zscore_threshold': params.get('zscore_threshold', 2.0),
            'zscore_exit': params.get('zscore_exit', 0.0),
            'rolling_window': params.get('rolling_window', 60),
            'max_holding_days': params.get('max_holding_days', 100)
        },
        'metadata': {
            'version': frozen['version'],
            'frozen_at': frozen['frozen_at'],
            'source': frozen['source']
        }
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    
    print(f"YAML config saved to {yaml_path}")
    
    # Update current symlink
    current_link = frozen_dir / "current.yaml"
    if current_link.exists():
        current_link.unlink()
    current_link.symlink_to(yaml_path.name)
    print(f"Updated current symlink to {yaml_path.name}")
    
    # Generate freeze report
    report = f"""# Parameter Freeze Report

## Freeze Information
- **Version**: {frozen['version']}
- **Week**: {week_id}
- **Frozen At**: {frozen['frozen_at']}
- **Valid Period**: {frozen['valid_from']} to {frozen['valid_to']}
- **Source**: {frozen['source']}
- **Hash**: {frozen['hash']}

## Frozen Parameters
```yaml
{yaml.dump(params, default_flow_style=False)}
```

## Deployment Status
- **Approved**: {'✅' if frozen['metadata']['approved'] else '❌'}
- **Deployed**: {'✅' if frozen['metadata']['deployed'] else '❌'}

## Files Generated
- JSON: `{json_path}`
- YAML: `{yaml_path}`
- Current: `{current_link}`

## Next Steps
1. Review parameters for production suitability
2. Run validation: `python scripts/validate.py --config {yaml_path}`
3. Deploy to production: Update live config to use frozen params
4. Monitor performance during the week
"""
    
    report_path = frozen_dir / f"FREEZE_REPORT_{week_id}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Freeze report saved to {report_path}")
    
    return frozen


def list_frozen_params():
    """List all frozen parameter sets."""
    frozen_dir = Path("configs/frozen")
    if not frozen_dir.exists():
        print("No frozen parameters found")
        return
    
    params_files = sorted(frozen_dir.glob("params_*.json"))
    
    if not params_files:
        print("No frozen parameters found")
        return
    
    print("\nFrozen Parameter Sets:")
    print("-" * 60)
    
    for filepath in params_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        current = " [CURRENT]" if (frozen_dir / "current.yaml").resolve() == (frozen_dir / f"params_{data['week_id']}.yaml").resolve() else ""
        deployed = " [DEPLOYED]" if data['metadata'].get('deployed') else ""
        
        print(f"Week {data['week_id']}: v{data['version']}{current}{deployed}")
        print(f"  Source: {data['source']}")
        print(f"  Frozen: {data['frozen_at'][:10]}")
        print(f"  Params: z_enter={data['parameters'].get('zscore_threshold', 'N/A')}, "
              f"z_exit={data['parameters'].get('zscore_exit', 'N/A')}, "
              f"window={data['parameters'].get('rolling_window', 'N/A')}")
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Freeze parameters for production")
    parser.add_argument("--source", choices=["auto", "optuna", "wfa", "manual"], 
                       default="auto", help="Parameter source")
    parser.add_argument("--list", action="store_true", help="List frozen params")
    parser.add_argument("--params", type=str, help="Manual params as JSON string")
    
    args = parser.parse_args()
    
    if args.list:
        list_frozen_params()
    else:
        override_params = None
        if args.params:
            override_params = json.loads(args.params)
        
        freeze_parameters(source=args.source, override_params=override_params)


if __name__ == "__main__":
    main()