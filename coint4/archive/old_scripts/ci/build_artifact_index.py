#!/usr/bin/env python3
"""Build unified artifact index for all generated files."""

import json
import hashlib
from pathlib import Path
from datetime import datetime
import pandas as pd


def calculate_file_hash(filepath: Path) -> str:
    """Calculate MD5 hash of file."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return "N/A"


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                              capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except:
        return 'unknown'


def get_artifact_metadata(filepath: Path) -> dict:
    """Extract extended metadata from artifact file.
    
    Args:
        filepath: Path to artifact file
        
    Returns:
        Dictionary with metadata fields
    """
    metadata = {}
    
    # Try to extract seed from filename or content
    if 'seed' in filepath.name.lower():
        # Extract seed from filename if present
        import re
        match = re.search(r'seed[_-]?(\d+)', filepath.name)
        if match:
            metadata['seed'] = int(match.group(1))
    
    # Check for locked params reference
    if 'locked' in str(filepath) or 'params' in filepath.name:
        metadata['params_locked_ref'] = str(filepath.stem)
    
    # Check for performance budget hits
    if filepath.suffix == '.json':
        try:
            with open(filepath) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Check for seed in content
                    if 'seed' in data:
                        metadata['seed'] = data['seed']
                    elif 'determinism' in data and 'seed' in data['determinism']:
                        metadata['seed'] = data['determinism']['seed']
                    
                    # Check for budget violations
                    if 'budget_exceeded' in data or 'perf_budget_hit' in data:
                        metadata['perf_budget_hit'] = True
        except:
            pass
    
    return metadata


def scan_artifacts(base_path: str = "artifacts") -> list:
    """Scan all artifacts and build index."""
    artifacts = []
    base = Path(base_path)
    
    if not base.exists():
        print(f"Creating artifacts directory: {base}")
        base.mkdir(parents=True, exist_ok=True)
        return artifacts
    
    # Define artifact categories
    categories = {
        'wfa': 'Walk-Forward Analysis',
        'optuna': 'Optuna Optimization',
        'live': 'Live Trading',
        'stress': 'Stress Testing',
        'deploy': 'Deployment',
        'traces': 'Debug Traces',
        'metrics': 'Performance Metrics'
    }
    
    # Scan all files
    for filepath in base.rglob("*"):
        if filepath.is_file():
            # Determine category
            rel_path = filepath.relative_to(base)
            category = rel_path.parts[0] if rel_path.parts else 'uncategorized'
            category_name = categories.get(category, category.title())
            
            # Get file info
            stat = filepath.stat()
            
            # Get extended metadata
            metadata = get_artifact_metadata(filepath)
            
            artifact = {
                'type': category,
                'path': str(filepath),
                'relative_path': str(rel_path),
                'category': category,
                'category_name': category_name,
                'filename': filepath.name,
                'extension': filepath.suffix,
                'size_bytes': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'hash': calculate_file_hash(filepath) if stat.st_size < 10 * 1024 * 1024 else "too_large",
                'git_commit': get_git_commit(),
                'config_hash': 'N/A',  # Would compute from config if available
                'seed': metadata.get('seed'),
                'params_locked_ref': metadata.get('params_locked_ref'),
                'data_lock_ref': metadata.get('data_lock_ref'),
                'perf_budget_hit': metadata.get('perf_budget_hit', False)
            }
            
            # Add metadata based on file type
            if filepath.suffix == '.json':
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            artifact['json_keys'] = list(data.keys())[:10]
                except:
                    artifact['json_keys'] = []
            
            elif filepath.suffix == '.csv':
                try:
                    df = pd.read_csv(filepath, nrows=1)
                    artifact['csv_columns'] = list(df.columns)
                    artifact['csv_rows'] = len(pd.read_csv(filepath))
                except:
                    artifact['csv_columns'] = []
                    artifact['csv_rows'] = 0
            
            elif filepath.suffix == '.md':
                artifact['type'] = 'report'
            
            artifacts.append(artifact)
    
    return artifacts


def generate_index_report(artifacts: list) -> str:
    """Generate markdown report of artifact index."""
    report = """# Artifact Registry Index

*Generated: {timestamp}*

## Summary

Total artifacts: {total}
Total size: {total_size_mb:.2f} MB

## By Category

""".format(
        timestamp=datetime.now().isoformat(),
        total=len(artifacts),
        total_size_mb=sum(a['size_mb'] for a in artifacts)
    )
    
    # Group by category
    from collections import defaultdict
    by_category = defaultdict(list)
    
    for artifact in artifacts:
        by_category[artifact['category_name']].append(artifact)
    
    # Add category sections
    for category, items in sorted(by_category.items()):
        report += f"### {category} ({len(items)} files)\n\n"
        report += "| File | Size | Modified | Type |\n"
        report += "|------|------|----------|------|\n"
        
        for item in sorted(items, key=lambda x: x['modified'], reverse=True)[:10]:
            report += f"| {item['filename']} | {item['size_mb']:.2f} MB | {item['modified'][:10]} | {item['extension']} |\n"
        
        if len(items) > 10:
            report += f"\n*...and {len(items) - 10} more files*\n"
        
        report += "\n"
    
    # Add recent files section
    report += "## Recent Files (Last 10)\n\n"
    report += "| File | Category | Modified | Size |\n"
    report += "|------|----------|----------|------|\n"
    
    recent = sorted(artifacts, key=lambda x: x['modified'], reverse=True)[:10]
    for item in recent:
        report += f"| {item['filename']} | {item['category_name']} | {item['modified'][:16]} | {item['size_mb']:.2f} MB |\n"
    
    # Add large files section
    report += "\n## Largest Files\n\n"
    report += "| File | Size | Category |\n"
    report += "|------|------|----------|\n"
    
    largest = sorted(artifacts, key=lambda x: x['size_mb'], reverse=True)[:5]
    for item in largest:
        report += f"| {item['filename']} | {item['size_mb']:.2f} MB | {item['category_name']} |\n"
    
    # Add file type distribution
    report += "\n## File Type Distribution\n\n"
    
    type_counts = defaultdict(int)
    type_sizes = defaultdict(float)
    
    for artifact in artifacts:
        ext = artifact['extension'] or 'no_extension'
        type_counts[ext] += 1
        type_sizes[ext] += artifact['size_mb']
    
    report += "| Type | Count | Total Size |\n"
    report += "|------|-------|------------|\n"
    
    for ext in sorted(type_counts.keys()):
        report += f"| {ext} | {type_counts[ext]} | {type_sizes[ext]:.2f} MB |\n"
    
    return report


def main():
    """Build and save artifact index."""
    print("Scanning artifacts...")
    artifacts = scan_artifacts()
    
    if not artifacts:
        print("No artifacts found")
        return
    
    print(f"Found {len(artifacts)} artifacts")
    
    # Save JSON index
    index_path = Path("artifacts/ARTIFACT_INDEX.json")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(index_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_artifacts': len(artifacts),
            'total_size_mb': sum(a['size_mb'] for a in artifacts),
            'artifacts': artifacts
        }, f, indent=2)
    
    print(f"Saved JSON index to {index_path}")
    
    # Save CSV index
    csv_path = Path("artifacts/ARTIFACT_INDEX.csv")
    df = pd.DataFrame(artifacts)
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV index to {csv_path}")
    
    # Generate markdown report
    report = generate_index_report(artifacts)
    report_path = Path("artifacts/ARTIFACT_REGISTRY.md")
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Saved registry report to {report_path}")
    
    # Print summary
    print("\nSummary:")
    print(f"  Total files: {len(artifacts)}")
    print(f"  Total size: {sum(a['size_mb'] for a in artifacts):.2f} MB")
    
    # Category breakdown
    from collections import Counter
    categories = Counter(a['category'] for a in artifacts)
    print("\nBy category:")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count} files")


if __name__ == "__main__":
    main()