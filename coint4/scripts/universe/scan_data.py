#!/usr/bin/env python3
"""Scan data inventory for available symbols and date ranges."""

import json
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import pyarrow.parquet as pq


def extract_symbols_fast(parquet_path: str, max_rows: int = 100_000) -> list:
    """Быстро достать уникальные символы из одного parquet:
       1) Если есть колонка 'symbol' — читаем только её из первого row group.
       2) Если её нет — возвращаем пустой список (широкий формат обработаем позже).
    """
    try:
        # Read the entire file, not just first row group
        tbl = pq.read_table(parquet_path, columns=['symbol'])
        ser = tbl['symbol'].to_pandas()
        
        # Limit if needed
        if len(ser) > max_rows:
            ser = ser.iloc[:max_rows]
        
        return sorted(pd.unique(ser.dropna()))
    except Exception:
        return []


def scan_data_inventory(data_root: str, timeframe: str = "15T", use_cache: bool = True):
    """Scan data directory for available symbols and date ranges.
    
    Args:
        data_root: Root directory with parquet files
        timeframe: Expected timeframe (for info only)
    
    Returns:
        dict with inventory info
    """
    data_path = Path(data_root)
    if not data_path.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    
    # Check for cached symbols
    cache_file = Path('artifacts/universe') / 'symbols_cache.json'
    if use_cache and cache_file.exists():
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            # Check if cache is recent (within 24 hours)
            cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2020-01-01'))
            if datetime.now(timezone.utc) - cache_time < timedelta(hours=24):
                print(f"📂 Using cached symbols from {cache_file}")
                return cache_data['inventory']
    
    # Find all parquet files
    parquet_files = list(data_path.glob("**/*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_root}")
    
    print(f"📊 Found {len(parquet_files)} parquet files")
    
    # Sample first and last files for date range
    # Using file paths to infer dates from year/month structure
    years = set()
    months = set()
    
    for pf in parquet_files:
        parts = pf.parts
        for i, part in enumerate(parts):
            if part.startswith("year="):
                years.add(int(part.split("=")[1]))
            elif part.startswith("month="):
                months.add(int(part.split("=")[1]))
    
    min_year = min(years) if years else 2021
    max_year = max(years) if years else 2025
    
    # Sample ALL files to get all unique symbols
    all_symbols = set()
    files_to_sample = len(parquet_files)  # Sample ALL files
    
    print(f"📖 Sampling {files_to_sample} parquet files for symbols...")
    for i, pf in enumerate(parquet_files):
        file_symbols = extract_symbols_fast(str(pf))
        if file_symbols:
            all_symbols.update(file_symbols)
            # Show progress every 10 files
            if (i + 1) % 10 == 0:
                print(f"  • Processed {i + 1}/{files_to_sample} files, found {len(all_symbols)} unique symbols so far...")
    
    symbols = sorted(list(all_symbols))
    
    if not symbols:
        print(f"⚠️ No symbols found (may be wide-format dataset)")
        symbols_note = "symbol column not found; dataset may be wide-formatted"
    else:
        symbols_note = None
    
    # Build inventory
    inventory = {
        "data_root": str(data_root),
        "timeframe_guess": timeframe,
        "min_ts": f"{min_year}-01-01T00:00:00",
        "max_ts": f"{max_year}-12-31T23:59:59",
        "symbols_count": len(symbols),
        "symbols_sample": symbols,  # Return all symbols, not just 20
        "total_files": len(parquet_files),
        "years_available": sorted(list(years)),
        "scan_timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Add note if symbols not found
    if symbols_note:
        inventory["symbols_note"] = symbols_note
    
    # Cache the results
    if use_cache:
        cache_file = Path('artifacts/universe') / 'symbols_cache.json'
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'inventory': inventory
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"💾 Cached {len(symbols)} symbols to {cache_file}")
    
    return inventory


def main():
    parser = argparse.ArgumentParser(description='Scan data inventory')
    parser.add_argument('--data-root', default='./data_downloaded',
                       help='Root directory with data')
    parser.add_argument('--timeframe', default='15T',
                       help='Expected timeframe')
    
    args = parser.parse_args()
    
    print(f"🔍 Scanning data inventory...")
    print(f"📁 Data root: {args.data_root}")
    
    try:
        inventory = scan_data_inventory(args.data_root, args.timeframe)
        
        # Create output directory
        out_dir = Path("artifacts/universe")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = out_dir / "DATA_SCAN.json"
        with open(json_path, 'w') as f:
            json.dump(inventory, f, indent=2)
        print(f"✅ Saved inventory to {json_path}")
        
        # Save Markdown report
        md_path = out_dir / "DATA_SCAN.md"
        with open(md_path, 'w') as f:
            symbols_section = chr(10).join(f'- {s}' for s in inventory['symbols_sample'])
            if 'symbols_note' in inventory:
                symbols_section = f"⚠️ {inventory['symbols_note']}\n\n" + symbols_section
            
            f.write(f"""# Data Inventory Scan Report

## Summary
- **Data Root**: `{inventory['data_root']}`
- **Timeframe**: {inventory['timeframe_guess']}
- **Date Range**: {inventory['min_ts']} to {inventory['max_ts']}
- **Total Files**: {inventory['total_files']}
- **Symbols Found**: {inventory['symbols_count']}
- **Years Available**: {', '.join(map(str, inventory['years_available']))}

## Sample Symbols (first 20)
{symbols_section}

## Scan Details
- **Scan Timestamp**: {inventory['scan_timestamp']}
- **Script**: `scripts/scan_data_inventory.py`

---
Generated by data inventory scanner
""")
        print(f"✅ Saved report to {md_path}")
        
        # Print summary
        print(f"\n📊 Inventory Summary:")
        print(f"  Date range: {inventory['min_ts'][:10]} to {inventory['max_ts'][:10]}")
        print(f"  Symbols: {inventory['symbols_count']}")
        print(f"  Files: {inventory['total_files']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())