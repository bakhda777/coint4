#!/usr/bin/env python3
"""
Analyze filter reasons CSV file to understand why pairs are being filtered out.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path
from collections import Counter


def analyze_filter_reasons(csv_path):
    """Analyze filter reasons and produce visualizations."""
    print(f"Analyzing filter reasons from: {csv_path}")
    
    # Load the data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} filter reason entries")
    
    # Count reasons
    reason_counts = Counter(df['reason'])
    total = len(df)
    
    # Convert to DataFrame for better visualization
    reasons_df = pd.DataFrame({
        'reason': list(reason_counts.keys()),
        'count': list(reason_counts.values()),
        'percentage': [count/total*100 for count in reason_counts.values()]
    })
    
    # Sort by count in descending order
    reasons_df = reasons_df.sort_values('count', ascending=False).reset_index(drop=True)
    
    # Print summary
    print("\nFilter Reason Summary:")
    for _, row in reasons_df.iterrows():
        print(f"{row['reason']}: {row['count']} pairs ({row['percentage']:.1f}%)")
    
    # Create visualization directory
    viz_dir = Path('results/visualizations')
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot the distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x='reason', y='percentage', data=reasons_df)
    plt.title('Distribution of Filter Reasons')
    plt.xlabel('Filter Reason')
    plt.ylabel('Percentage of Filtered Pairs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_path = viz_dir / f"filter_reasons_distribution_{os.path.basename(csv_path).replace('.csv', '')}.png"
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    
    # Analyze specific pairs
    # Count how many times each symbol appears in filtered pairs
    s1_counter = Counter(df['s1'])
    s2_counter = Counter(df['s2'])
    
    # Combine them to get total appearances
    all_symbols = set(s1_counter.keys()) | set(s2_counter.keys())
    symbol_counts = {symbol: s1_counter.get(symbol, 0) + s2_counter.get(symbol, 0) for symbol in all_symbols}
    
    # Convert to DataFrame for analysis
    symbols_df = pd.DataFrame({
        'symbol': list(symbol_counts.keys()),
        'filtered_count': list(symbol_counts.values())
    }).sort_values('filtered_count', ascending=False).reset_index(drop=True)
    
    # Print top filtered symbols
    print("\nTop 10 Most Filtered Symbols:")
    for _, row in symbols_df.head(10).iterrows():
        print(f"{row['symbol']}: filtered in {row['filtered_count']} pairs")
    
    # Plot top filtered symbols
    plt.figure(figsize=(12, 6))
    top_n = 20  # Show top 20 symbols
    top_symbols = symbols_df.head(top_n)
    sns.barplot(x='symbol', y='filtered_count', data=top_symbols)
    plt.title(f'Top {top_n} Most Filtered Symbols')
    plt.xlabel('Symbol')
    plt.ylabel('Number of Filtered Pairs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_path = viz_dir / f"top_filtered_symbols_{os.path.basename(csv_path).replace('.csv', '')}.png"
    plt.savefig(plot_path)
    print(f"Symbol plot saved to: {plot_path}")
    
    # Analyze symbol-reason relationship
    # Create a multi-index DataFrame for symbol + reason
    symbol_reason_counts = pd.DataFrame()
    
    # Process s1 symbols
    s1_reasons = df.groupby(['s1', 'reason']).size().reset_index(name='count')
    s1_reasons.columns = ['symbol', 'reason', 'count']
    
    # Process s2 symbols
    s2_reasons = df.groupby(['s2', 'reason']).size().reset_index(name='count')
    s2_reasons.columns = ['symbol', 'reason', 'count']
    
    # Combine
    symbol_reason_counts = pd.concat([s1_reasons, s2_reasons])
    symbol_reason_counts = symbol_reason_counts.groupby(['symbol', 'reason']).sum().reset_index()
    
    # Get top symbols
    top_symbols_list = symbols_df.head(10)['symbol'].tolist()
    
    # Filter for top symbols
    top_symbol_reasons = symbol_reason_counts[symbol_reason_counts['symbol'].isin(top_symbols_list)]
    
    # Plot heatmap for top symbols and their filter reasons
    plt.figure(figsize=(12, 8))
    pivot_data = top_symbol_reasons.pivot(index='symbol', columns='reason', values='count').fillna(0)
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd')
    plt.title('Filter Reasons by Top Symbols')
    plt.tight_layout()
    
    # Save the heatmap
    plot_path = viz_dir / f"symbol_reason_heatmap_{os.path.basename(csv_path).replace('.csv', '')}.png"
    plt.savefig(plot_path)
    print(f"Heatmap saved to: {plot_path}")
    
    return reasons_df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Find most recent filter reasons file
        results_dir = Path('results')
        filter_files = list(results_dir.glob('filter_reasons_*.csv'))
        if not filter_files:
            print("No filter_reasons_*.csv files found in results directory")
            sys.exit(1)
        
        # Sort by modification time, newest first
        filter_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        csv_path = str(filter_files[0])
        print(f"Using most recent filter reasons file: {csv_path}")
    
    analyze_filter_reasons(csv_path)