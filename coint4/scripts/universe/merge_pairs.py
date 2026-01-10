#!/usr/bin/env python3
"""Merge multiple pairs_universe.yaml files with deduplication."""

import argparse
import glob
from pathlib import Path
import yaml
import sys


def merge_pairs(glob_pattern, top_k=None, min_score=float('-inf')):
    """Merge pairs from multiple YAML files."""
    all_pairs = {}
    
    # Find all matching files
    files = glob.glob(glob_pattern)
    if not files:
        print(f"⚠️ No files found matching: {glob_pattern}")
        return []
    
    print(f"📂 Found {len(files)} files to merge:")
    for f in files:
        print(f"  • {f}")
    
    # Load and merge
    for filepath in files:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
            
        if not data or 'pairs' not in data:
            continue
            
        # Add pairs, keeping highest score for duplicates
        for pair_data in data['pairs']:
            pair_key = pair_data['pair']
            score = pair_data.get('score', 0)
            
            if pair_key not in all_pairs or score > all_pairs[pair_key]['score']:
                all_pairs[pair_key] = pair_data
    
    # Filter by min_score
    filtered = [p for p in all_pairs.values() if p.get('score', 0) >= min_score]
    
    # Sort by score descending
    sorted_pairs = sorted(filtered, key=lambda x: x.get('score', 0), reverse=True)
    
    # Apply top_k limit
    if top_k:
        sorted_pairs = sorted_pairs[:top_k]
    
    return sorted_pairs


def main():
    parser = argparse.ArgumentParser(description='Merge pairs from multiple universe selections')
    parser.add_argument('--glob', default='artifacts/universe/*/pairs_universe.yaml',
                       help='Glob pattern for input files')
    parser.add_argument('--out', default='bench/pairs_merged.yaml',
                       help='Output file path')
    parser.add_argument('--top-k', type=int, help='Keep only top K pairs')
    parser.add_argument('--min-score', type=float, default=float('-inf'),
                       help='Minimum score threshold')
    
    args = parser.parse_args()
    
    # Merge pairs
    merged = merge_pairs(args.glob, args.top_k, args.min_score)
    
    if not merged:
        print("❌ No pairs to merge")
        return 1
    
    print(f"\n📊 Merged statistics:")
    print(f"  Total unique pairs: {len(merged)}")
    if merged:
        scores = [p.get('score', 0) for p in merged]
        print(f"  Best score: {max(scores):.3f}")
        print(f"  Worst score: {min(scores):.3f}")
        print(f"  Average score: {sum(scores)/len(scores):.3f}")
    
    # Create output
    output = {
        'metadata': {
            'type': 'merged',
            'source_pattern': args.glob,
            'total_pairs': len(merged),
            'filters': {
                'top_k': args.top_k,
                'min_score': args.min_score
            }
        },
        'pairs': merged
    }
    
    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Saved {len(merged)} pairs to {out_path}")
    
    # Show top 5
    print("\n🏆 Top 5 pairs:")
    for i, pair in enumerate(merged[:5], 1):
        print(f"  {i}. {pair['pair']}: score={pair.get('score', 0):.3f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())