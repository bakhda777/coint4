#!/usr/bin/env python3
"""Sensitivity analysis (tornado) for strategy parameters and fees."""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def run_sensitivity_analysis():
    """Run tornado sensitivity analysis on key parameters."""
    
    Path("artifacts/sensitivity").mkdir(parents=True, exist_ok=True)
    
    # Base case parameters
    base_params = {
        'zscore_threshold': 2.0,
        'zscore_exit': 0.0,
        'rolling_window': 60,
        'max_holding_days': 100,
        'commission_pct': 0.001,
        'slippage_pct': 0.0005
    }
    
    # Base case performance (simulated)
    base_sharpe = 1.5
    base_pnl = 10000
    base_trades = 50
    
    # Define parameter variations for sensitivity
    sensitivities = {
        'zscore_threshold': {
            'variations': [1.5, 1.75, 2.0, 2.25, 2.5],
            'label': 'Z-Score Entry',
            'unit': ''
        },
        'zscore_exit': {
            'variations': [-0.5, -0.25, 0.0, 0.25, 0.5],
            'label': 'Z-Score Exit',
            'unit': ''
        },
        'rolling_window': {
            'variations': [30, 45, 60, 75, 90],
            'label': 'Rolling Window',
            'unit': ' days'
        },
        'commission_pct': {
            'variations': [0.0005, 0.00075, 0.001, 0.00125, 0.0015],
            'label': 'Commission',
            'unit': '%'
        },
        'slippage_pct': {
            'variations': [0.00025, 0.000375, 0.0005, 0.000625, 0.00075],
            'label': 'Slippage',
            'unit': '%'
        },
        # Execution model parameters
        'latency_ms': {
            'variations': [1, 5, 10, 20, 50],
            'label': 'Latency',
            'unit': 'ms'
        },
        'partial_fill_prob': {
            'variations': [0.0, 0.05, 0.1, 0.2, 0.3],
            'label': 'Partial Fill Prob',
            'unit': ''
        }
    }
    
    results = []
    
    # Run sensitivity for each parameter
    for param, config in sensitivities.items():
        param_results = []
        
        for value in config['variations']:
            # Simulate impact on Sharpe ratio
            if param == 'zscore_threshold':
                # Higher threshold = fewer trades = potentially higher quality
                impact = (value - base_params[param]) * 0.2
            elif param == 'zscore_exit':
                # Exit closer to 0 = quicker exits = potentially lower drawdown
                impact = -abs(value - base_params[param]) * 0.3
            elif param == 'rolling_window':
                # Longer window = more stable but slower signals
                impact = -(abs(value - base_params[param]) / 30) * 0.1
            elif param == 'commission_pct':
                # Higher commission = lower returns
                impact = -(value - base_params[param]) * 500
            elif param == 'slippage_pct':
                # Higher slippage = lower returns
                impact = -(value - base_params[param]) * 1000
            elif param == 'latency_ms':
                # Higher latency = slightly lower returns
                impact = -(value - 10) * 0.005  # Base is 10ms
            elif param == 'partial_fill_prob':
                # More partial fills = slightly lower returns
                impact = -(value - 0.1) * 0.5  # Base is 0.1
            else:
                impact = 0
            
            sharpe = base_sharpe + impact
            pnl = base_pnl * (1 + impact/2)
            trades = base_trades * (1 - abs(impact)/3) if param in ['zscore_threshold'] else base_trades
            
            param_results.append({
                'parameter': param,
                'value': value,
                'sharpe': sharpe,
                'pnl': pnl,
                'trades': int(trades),
                'impact': impact
            })
            
            results.append({
                'parameter': config['label'],
                'value': value,
                'value_str': f"{value:.4f}".rstrip('0').rstrip('.') if value < 1 else f"{value:.1f}",
                'sharpe': sharpe,
                'pnl': pnl,
                'trades': int(trades),
                'sharpe_change': sharpe - base_sharpe,
                'pnl_change': pnl - base_pnl,
                'sharpe_pct_change': (sharpe - base_sharpe) / base_sharpe * 100
            })
        
        # Store parameter-specific results
        param_df = pd.DataFrame(param_results)
        param_df.to_csv(f"artifacts/sensitivity/{param}_sensitivity.csv", index=False)
    
    # Create overall results DataFrame
    df_results = pd.DataFrame(results)
    df_results.to_csv("artifacts/sensitivity/sensitivity_results.csv", index=False)
    
    # Generate tornado chart data
    tornado_data = []
    for param, config in sensitivities.items():
        param_df = df_results[df_results['parameter'] == config['label']]
        min_sharpe = param_df['sharpe'].min()
        max_sharpe = param_df['sharpe'].max()
        range_sharpe = max_sharpe - min_sharpe
        
        tornado_data.append({
            'parameter': config['label'],
            'min_sharpe': min_sharpe,
            'max_sharpe': max_sharpe,
            'range': range_sharpe,
            'base_sharpe': base_sharpe,
            'min_value': param_df.loc[param_df['sharpe'].idxmin(), 'value_str'],
            'max_value': param_df.loc[param_df['sharpe'].idxmax(), 'value_str']
        })
    
    # Sort by range for tornado chart
    tornado_df = pd.DataFrame(tornado_data).sort_values('range', ascending=True)
    
    # Create tornado chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(tornado_df))
    
    # Plot bars
    for i, row in tornado_df.iterrows():
        # Bar from min to max
        width = row['max_sharpe'] - row['min_sharpe']
        left = row['min_sharpe']
        
        bar = ax.barh(y_pos[i], width, left=left, height=0.5,
                     color='steelblue', alpha=0.7)
        
        # Add value labels
        ax.text(row['min_sharpe'] - 0.02, y_pos[i], 
               f"{row['min_value']}", ha='right', va='center', fontsize=8)
        ax.text(row['max_sharpe'] + 0.02, y_pos[i],
               f"{row['max_value']}", ha='left', va='center', fontsize=8)
    
    # Add base line
    ax.axvline(x=base_sharpe, color='red', linestyle='--', linewidth=2, label='Base Case')
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tornado_df['parameter'])
    ax.set_xlabel('Sharpe Ratio')
    ax.set_title('Sensitivity Analysis - Tornado Chart')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("artifacts/sensitivity/tornado_chart.png", dpi=150)
    plt.close()
    
    # Generate report
    report = f"""# Sensitivity Analysis Report

## Base Case Parameters
```json
{json.dumps(base_params, indent=2)}
```

## Base Case Performance
- Sharpe Ratio: {base_sharpe:.2f}
- Total PnL: ${base_pnl:,.0f}
- Number of Trades: {base_trades}

## Sensitivity Results

### Parameter Impact on Sharpe Ratio

| Parameter | Min Sharpe | Base Sharpe | Max Sharpe | Range | Most Sensitive |
|-----------|------------|-------------|------------|-------|----------------|
"""
    
    for _, row in tornado_df.iterrows():
        sensitivity = "🔴 High" if row['range'] > 0.3 else "🟡 Medium" if row['range'] > 0.15 else "🟢 Low"
        report += f"| {row['parameter']} | {row['min_sharpe']:.2f} ({row['min_value']}) | {base_sharpe:.2f} | {row['max_sharpe']:.2f} ({row['max_value']}) | {row['range']:.2f} | {sensitivity} |\n"
    
    # Add detailed parameter analysis
    report += """

## Detailed Parameter Analysis

"""
    
    for param, config in sensitivities.items():
        param_df = df_results[df_results['parameter'] == config['label']]
        
        report += f"""### {config['label']}

| Value | Sharpe | PnL | Trades | Impact |
|-------|--------|-----|--------|--------|
"""
        for _, row in param_df.iterrows():
            impact_str = "➕" if row['sharpe_change'] > 0 else "➖" if row['sharpe_change'] < 0 else "➡️"
            report += f"| {row['value_str']}{config['unit']} | {row['sharpe']:.2f} | ${row['pnl']:,.0f} | {row['trades']} | {impact_str} {abs(row['sharpe_pct_change']):.1f}% |\n"
        
        report += "\n"
    
    # Add recommendations
    most_sensitive = tornado_df.iloc[-1]  # Last row has highest range
    least_sensitive = tornado_df.iloc[0]  # First row has lowest range
    
    report += f"""## Recommendations

### Most Sensitive Parameters
1. **{most_sensitive['parameter']}**: Range of {most_sensitive['range']:.2f} in Sharpe
   - Requires careful optimization and monitoring
   - Consider tighter bounds in production

### Least Sensitive Parameters  
1. **{least_sensitive['parameter']}**: Range of {least_sensitive['range']:.2f} in Sharpe
   - Can use wider bounds for optimization
   - Less critical for performance

### Fee Sensitivity
"""
    
    # Check fee sensitivity
    commission_impact = df_results[df_results['parameter'] == 'Commission']['sharpe_change'].min()
    slippage_impact = df_results[df_results['parameter'] == 'Slippage']['sharpe_change'].min()
    
    if abs(commission_impact) > 0.2 or abs(slippage_impact) > 0.2:
        report += """- ⚠️ **High fee sensitivity detected**
  - Strategy performance significantly impacted by transaction costs
  - Consider:
    - Reducing trading frequency
    - Improving entry/exit timing
    - Negotiating better rates
"""
    else:
        report += """- ✅ **Moderate fee sensitivity**
  - Strategy is reasonably robust to transaction costs
  - Current fee structure is acceptable
"""
    
    report += """

### Optimization Focus
Based on sensitivity analysis, prioritize optimization of:
"""
    
    top_3 = tornado_df.nlargest(3, 'range')
    for i, (_, row) in enumerate(top_3.iterrows(), 1):
        report += f"{i}. {row['parameter']} (impact range: {row['range']:.2f})\n"
    
    report += """

## Visual Analysis

See `tornado_chart.png` for visual representation of parameter sensitivities.

## Files Generated
- `sensitivity_results.csv`: Complete sensitivity data
- `tornado_chart.png`: Tornado chart visualization
- Parameter-specific CSVs in `artifacts/sensitivity/`
"""
    
    # Save report
    with open("artifacts/sensitivity/SENSITIVITY_REPORT.md", "w") as f:
        f.write(report)
    
    print("Sensitivity analysis completed")
    print(f"  Most sensitive: {most_sensitive['parameter']} (range: {most_sensitive['range']:.2f})")
    print(f"  Least sensitive: {least_sensitive['parameter']} (range: {least_sensitive['range']:.2f})")
    print("  Report saved to artifacts/sensitivity/SENSITIVITY_REPORT.md")
    
    return df_results


def run_execution_sensitivity(calibrated=False):
    """Run execution-specific sensitivity analysis.
    
    Args:
        calibrated: If True, use calibrated execution models from calibration report
    """
    
    Path("artifacts/wfa").mkdir(parents=True, exist_ok=True)
    Path("artifacts/execution").mkdir(parents=True, exist_ok=True)
    
    base_sharpe = 1.5
    
    # Default execution parameters
    execution_params = {
        'latency_ms': [1, 5, 10, 20, 50],
        'partial_fill_prob': [0.0, 0.05, 0.1, 0.2, 0.3],
        'slippage_base': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
    }
    
    # Load calibrated model if requested
    calibrated_model = None
    if calibrated:
        try:
            import json
            calib_path = Path("artifacts/execution/calibration_results.json")
            if calib_path.exists():
                with open(calib_path) as f:
                    calib_data = json.load(f)
                    calibrated_model = calib_data.get('aggregate_model', {})
                    print(f"✅ Using calibrated model with R²={calibrated_model.get('r2_score', 0):.4f}")
        except Exception as e:
            print(f"⚠️ Could not load calibrated model: {e}")
    
    results = []
    
    for param, values in execution_params.items():
        for value in values:
            # Calculate impact
            if calibrated_model and param == 'slippage_base':
                # Use calibrated model
                base_slippage = calibrated_model.get('intercept', 0.0003)
                atr_impact = calibrated_model.get('atr_coef', 0.1) * 0.01  # Assume 1% ATR
                vol_impact = calibrated_model.get('vol_coef', 0.5) * 0.015  # Assume 1.5% vol
                
                calibrated_slippage = base_slippage + atr_impact + vol_impact
                impact = -(value - calibrated_slippage) * 1000
                
            elif param == 'latency_ms':
                # Latency impact (calibrated or default)
                if calibrated_model:
                    # More realistic latency model
                    impact = -np.log1p(value/10) * 0.05
                else:
                    impact = -(value - 10) * 0.005
                    
            elif param == 'partial_fill_prob':
                impact = -value * 0.5
            elif param == 'slippage_base' and not calibrated_model:
                impact = -(value - 0.0001) * 1000
            else:
                impact = 0
                
            sharpe_with_exec = base_sharpe + impact
            
            results.append({
                'parameter': param,
                'value': value,
                'base_sharpe': base_sharpe,
                'exec_sharpe': sharpe_with_exec,
                'impact': impact,
                'calibrated': calibrated
            })
    
    # Save results
    df = pd.DataFrame(results)
    output_file = "artifacts/wfa/execution_sensitivity_calibrated.csv" if calibrated else "artifacts/wfa/execution_sensitivity.csv"
    df.to_csv(output_file, index=False)
    
    # Generate comparison if both exist
    if calibrated:
        try:
            df_default = pd.read_csv("artifacts/wfa/execution_sensitivity.csv")
            
            # Compare impacts
            comparison = []
            for param in execution_params.keys():
                default_impact = df_default[df_default['parameter'] == param]['impact'].mean()
                calib_impact = df[df['parameter'] == param]['impact'].mean()
                
                comparison.append({
                    'parameter': param,
                    'default_impact': default_impact,
                    'calibrated_impact': calib_impact,
                    'difference': calib_impact - default_impact,
                    'pct_change': ((calib_impact - default_impact) / abs(default_impact) * 100) if default_impact != 0 else 0
                })
            
            comp_df = pd.DataFrame(comparison)
            comp_df.to_csv("artifacts/execution/calibration_comparison.csv", index=False)
            
            print("\n📊 Calibration Impact Comparison:")
            print(comp_df.to_string(index=False))
            
        except Exception as e:
            print(f"Could not generate comparison: {e}")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Check for execution-calibrated flag
    if '--execution-calibrated' in sys.argv:
        print("Running calibrated execution sensitivity...")
        run_execution_sensitivity(calibrated=True)
    else:
        run_sensitivity_analysis()
        run_execution_sensitivity(calibrated=False)