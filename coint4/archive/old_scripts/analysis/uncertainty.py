#!/usr/bin/env python3
"""
Run uncertainty analysis with bootstrap confidence intervals.
Generates confidence intervals for Sharpe/PSR/DSR metrics.
"""

import sys
import argparse
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from coint2.stats.bootstrap import (
    MetricsBootstrap,
    load_oos_returns_from_wfa,
    save_confidence_report
)


def load_portfolio_weights(weights_file: str) -> Optional[Dict[str, float]]:
    """Load portfolio weights from CSV file."""
    weights_path = Path(weights_file)
    
    if not weights_path.exists():
        print(f"⚠️ Portfolio weights not found at {weights_file}")
        return None
    
    try:
        weights_df = pd.read_csv(weights_path)
        
        if 'pair' in weights_df.columns and 'weight' in weights_df.columns:
            weights_dict = dict(zip(weights_df['pair'], weights_df['weight']))
            print(f"✅ Loaded {len(weights_dict)} portfolio weights")
            return weights_dict
        else:
            print(f"⚠️ Invalid weights file format: missing 'pair' or 'weight' columns")
            return None
            
    except Exception as e:
        print(f"⚠️ Error loading portfolio weights: {e}")
        return None


def integrate_into_wfa_report(
    confidence_results: Dict,
    wfa_report_path: str = 'artifacts/wfa/WFA_REPORT.md'
) -> None:
    """Integrate confidence summary into WFA report."""
    wfa_path = Path(wfa_report_path)
    
    if not wfa_path.exists():
        print(f"⚠️ WFA report not found at {wfa_report_path}")
        return
    
    # Read existing report
    with open(wfa_path, 'r') as f:
        content = f.read()
    
    # Create confidence summary
    confidence_section = "\n\n## Portfolio Confidence Analysis\n"
    
    if 'PORTFOLIO' in confidence_results:
        portfolio = confidence_results['PORTFOLIO']
        
        # Sharpe confidence
        if 'sharpe' in portfolio:
            sharpe = portfolio['sharpe']
            confidence_section += f"- **Sharpe 90% CI**: [{sharpe.get('p05', 0):.2f}, {sharpe.get('p95', 0):.2f}] (observed: {sharpe.get('observed', 0):.2f})\n"
        
        # PSR confidence
        if 'psr' in portfolio:
            psr = portfolio['psr']
            confidence_section += f"- **PSR 90% CI**: [{psr.get('p05', 0):.3f}, {psr.get('p95', 0):.3f}] (observed: {psr.get('observed', 0):.3f})\n"
        
        # Risk assessment
        sharpe_p05 = portfolio.get('sharpe', {}).get('p05', 0)
        psr_p05 = portfolio.get('psr', {}).get('p05', 0)
        
        if sharpe_p05 > 0.6 and psr_p05 > 0.90:
            confidence_section += "- **Risk Level**: ✅ LOW (Strong confidence bounds)\n"
        elif sharpe_p05 > 0.3 and psr_p05 > 0.70:
            confidence_section += "- **Risk Level**: ⚠️ MODERATE (Acceptable bounds)\n"
        else:
            confidence_section += "- **Risk Level**: 🚨 HIGH (Weak confidence bounds)\n"
    
    confidence_section += f"\n*Full analysis: [CONFIDENCE_REPORT.md](../uncertainty/CONFIDENCE_REPORT.md)*\n"
    
    # Add section before any existing "## " section or append to end
    if '\n## ' in content:
        # Find first section and insert before it
        section_pos = content.find('\n## ')
        new_content = content[:section_pos] + confidence_section + content[section_pos:]
    else:
        # Append to end
        new_content = content + confidence_section
    
    # Write updated report
    with open(wfa_path, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Updated WFA report with confidence summary")


def run_uncertainty_analysis(
    wfa_results_path: str = 'artifacts/wfa/results_per_fold.csv',
    portfolio_weights_path: str = 'artifacts/portfolio/weights.csv',
    output_dir: str = 'artifacts/uncertainty',
    config: Optional[Dict] = None
) -> None:
    """Run complete uncertainty analysis.
    
    Args:
        wfa_results_path: Path to WFA results CSV
        portfolio_weights_path: Path to portfolio weights CSV
        output_dir: Output directory for results
        config: Bootstrap configuration
    """
    print("=" * 60)
    print("UNCERTAINTY ANALYSIS - Bootstrap Confidence Intervals")
    print("=" * 60)
    
    # Default configuration
    if config is None:
        config = {
            'block_size': 20,
            'n_bootstrap': 1000,
            'confidence_levels': [0.05, 0.50, 0.95],
            'seed': 42
        }
    
    print(f"\n📊 Configuration:")
    print(f"   Block size: {config['block_size']}")
    print(f"   Bootstrap samples: {config['n_bootstrap']}")
    print(f"   Confidence levels: {config['confidence_levels']}")
    
    # Initialize bootstrap analyzer
    bootstrap = MetricsBootstrap(
        block_size=config['block_size'],
        n_bootstrap=config['n_bootstrap'],
        confidence_levels=config['confidence_levels'],
        seed=config['seed']
    )
    
    # Load OOS returns
    print(f"\n📈 Loading OOS returns from {wfa_results_path}...")
    returns_data = load_oos_returns_from_wfa(wfa_results_path)
    print(f"   Loaded {len(returns_data)} observations for {len(returns_data.columns)} pairs")
    
    # Load portfolio weights
    print(f"\n💼 Loading portfolio weights from {portfolio_weights_path}...")
    portfolio_weights = load_portfolio_weights(portfolio_weights_path)
    
    # Run uncertainty analysis
    print(f"\n🔄 Running bootstrap analysis...")
    confidence_results = bootstrap.analyze_portfolio_uncertainty(
        returns_data, portfolio_weights
    )
    
    print(f"   Analyzed {len(confidence_results)} entities")
    
    # Save results
    print(f"\n💾 Saving results to {output_dir}...")
    md_path, csv_path = save_confidence_report(
        confidence_results, output_dir, config
    )
    
    print(f"   ✅ Markdown report: {md_path}")
    print(f"   ✅ CSV data: {csv_path}")
    
    # Integrate into WFA report
    print(f"\n🔗 Integrating into WFA report...")
    integrate_into_wfa_report(confidence_results)
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if 'PORTFOLIO' in confidence_results:
        portfolio = confidence_results['PORTFOLIO']
        
        print(f"\n💼 Portfolio Confidence:")
        if 'sharpe' in portfolio:
            sharpe = portfolio['sharpe']
            print(f"   Sharpe [P05, P95]: [{sharpe.get('p05', 0):.2f}, {sharpe.get('p95', 0):.2f}]")
        
        if 'psr' in portfolio:
            psr = portfolio['psr']
            print(f"   PSR [P05, P95]: [{psr.get('p05', 0):.3f}, {psr.get('p95', 0):.3f}]")
        
        # Risk assessment
        sharpe_p05 = portfolio.get('sharpe', {}).get('p05', 0)
        psr_p05 = portfolio.get('psr', {}).get('p05', 0)
        
        print(f"\n🎯 Risk Assessment:")
        if sharpe_p05 > 0.6 and psr_p05 > 0.90:
            print(f"   Status: ✅ LOW RISK")
        elif sharpe_p05 > 0.3 and psr_p05 > 0.70:
            print(f"   Status: ⚠️ MODERATE RISK")
        else:
            print(f"   Status: 🚨 HIGH RISK")
        
        print(f"   Sharpe P05: {sharpe_p05:.3f}")
        print(f"   PSR P05: {psr_p05:.3f}")
    
    print(f"\n📊 Analysis complete. Check {md_path} for full report.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run uncertainty analysis with bootstrap confidence intervals')
    
    parser.add_argument('--wfa-results', default='artifacts/wfa/results_per_fold.csv',
                       help='Path to WFA results CSV')
    parser.add_argument('--portfolio-weights', default='artifacts/portfolio/weights.csv',
                       help='Path to portfolio weights CSV')
    parser.add_argument('--output-dir', default='artifacts/uncertainty',
                       help='Output directory for results')
    parser.add_argument('--config', type=str,
                       help='Path to bootstrap configuration YAML')
    parser.add_argument('--block-size', type=int, default=20,
                       help='Bootstrap block size (10-50)')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                       help='Number of bootstrap samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (fewer bootstrap samples)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
    
    # Override with command line args
    if config is None:
        config = {
            'confidence_levels': [0.05, 0.50, 0.95]
        }
    
    config.update({
        'block_size': args.block_size,
        'n_bootstrap': 100 if args.quick else args.n_bootstrap,
        'seed': args.seed
    })
    
    # Run analysis
    run_uncertainty_analysis(
        wfa_results_path=args.wfa_results,
        portfolio_weights_path=args.portfolio_weights,
        output_dir=args.output_dir,
        config=config
    )


if __name__ == '__main__':
    main()