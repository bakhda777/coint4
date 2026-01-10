#!/usr/bin/env python3
"""Validate YAML files for correctness and completeness."""

import yaml
import argparse
from pathlib import Path
import sys


def validate_pairs_yaml(filepath):
    """Validate pairs YAML file structure and content."""
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        print(f"💡 Recommendation: Run aggregation first:")
        print(f"   python scripts/aggregate_universe.py --min-windows 1")
        return False
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Check for common issues
        if 'EOF' in content and '<' in content:
            print(f"❌ File contains invalid EOF markers")
            print(f"💡 Clean the file and re-run aggregation")
            return False
            
        # Parse YAML
        data = yaml.safe_load(content)
        
        if not data:
            print(f"❌ File is empty or invalid YAML")
            print(f"💡 Recommendations:")
            print(f"   1. Re-run universe selection with relaxed criteria")
            print(f"   2. Check if data period has enough symbols")
            print(f"   3. Lower min_windows threshold in aggregation")
            return False
        
        # Check required fields
        if 'pairs' not in data:
            print(f"❌ Missing 'pairs' field in YAML")
            return False
            
        pairs = data.get('pairs', [])
        
        if not pairs:
            print(f"⚠️ No pairs found in file!")
            print(f"💡 Recommendations:")
            print(f"   1. Re-run with relaxed criteria:")
            print(f"      - Increase coint_pvalue_max to 0.20")
            print(f"      - Decrease min_cross to 5")
            print(f"      - Expand hurst range to 0.1-0.8")
            print(f"   2. Use fewer windows or lower min_windows")
            print(f"   3. Check data availability for the period")
            return False
        
        # Validate each pair
        for i, pair in enumerate(pairs):
            if not isinstance(pair, dict):
                print(f"❌ Pair {i} is not a dictionary")
                return False
                
            if 'pair' not in pair:
                print(f"❌ Pair {i} missing 'pair' field")
                return False
                
            if 'weight' not in pair:
                print(f"⚠️ Pair {i} missing 'weight' field (will use default)")
        
        # Check metadata
        metadata = data.get('metadata', {})
        if metadata:
            print(f"📊 Metadata found:")
            print(f"   - Generated: {metadata.get('generated', 'N/A')}")
            print(f"   - Windows: {metadata.get('total_windows_analyzed', 'N/A')}")
            print(f"   - Selected: {metadata.get('total_pairs_selected', len(pairs))}")
        
        # Validate weights sum
        total_weight = sum(p.get('weight', 1.0) for p in pairs)
        if abs(total_weight - 1.0) > 0.01 and abs(total_weight - len(pairs)) > 0.01:
            print(f"⚠️ Weights sum to {total_weight:.3f} (expected 1.0 or {len(pairs)})")
        
        print(f"✅ Valid YAML with {len(pairs)} pairs")
        
        # Show sample
        if pairs:
            print(f"\n📋 First 5 pairs:")
            for p in pairs[:5]:
                print(f"   - {p['pair']}: weight={p.get('weight', 1.0):.3f}")
        
        return True
        
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML syntax: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Validate YAML files')
    parser.add_argument('filepath', help='Path to YAML file to validate')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed validation info')
    
    args = parser.parse_args()
    
    print(f"🔍 Validating {args.filepath}...")
    
    if validate_pairs_yaml(args.filepath):
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())