#!/usr/bin/env python3
"""Run multiple universe selection jobs from config."""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

import yaml


def run_job(job: dict, data_config: dict) -> int:
    """Run single selection job."""
    name = job['name']
    print(f"\n{'='*60}")
    print(f"🚀 Running job: {name}")
    print(f"{'='*60}")
    
    # Prepare output directory
    out_dir = Path(f"artifacts/universe/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save criteria to temp file
    criteria_file = out_dir / 'criteria.yaml'
    with open(criteria_file, 'w') as f:
        yaml.dump(job['criteria'], f)
    
    # Build command
    cmd = [
        sys.executable,
        'scripts/universe/select_pairs.py',
        '--data-root', data_config['root'],
        '--timeframe', data_config['timeframe'],
        '--period-start', job['period']['start'],
        '--period-end', job['period']['end'],
        '--criteria-config', str(criteria_file),
        '--out-dir', str(out_dir),
        '--top-n', str(job['top_n']),
        '--limit-pairs', str(job.get('limit_pairs', 1000))
    ]
    
    if job.get('diversify_by_base', True):
        cmd.append('--diversify-by-base')
        cmd.extend(['--max-per-base', str(job.get('max_per_base', 5))])
    
    # Run command
    print(f"📋 Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        
        # Save job results summary
        summary_file = out_dir / 'job_summary.yaml'
        with open(summary_file, 'w') as f:
            yaml.dump({
                'job': job,
                'data': data_config,
                'executed_at': datetime.now(timezone.utc).isoformat(),
                'status': 'SUCCESS'
            }, f)
        
        print(f"✅ Job {name} completed successfully")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Job {name} failed with code {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        
        # Save error summary
        error_file = out_dir / 'job_error.yaml'
        with open(error_file, 'w') as f:
            yaml.dump({
                'job': job,
                'error': str(e),
                'stderr': e.stderr,
                'executed_at': datetime.now(timezone.utc).isoformat(),
                'status': 'FAILED'
            }, f)
        
        return e.returncode


def main():
    parser = argparse.ArgumentParser(description='Run universe selection jobs')
    parser.add_argument('--config', default='configs/universe_jobs.yaml',
                       help='Jobs configuration file')
    parser.add_argument('--jobs', nargs='*',
                       help='Specific job names to run (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print jobs without running')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    jobs = config['jobs']
    
    # Filter jobs if specified
    if args.jobs:
        jobs = [j for j in jobs if j['name'] in args.jobs]
        if not jobs:
            print(f"❌ No matching jobs found for: {args.jobs}")
            return 1
    
    print(f"📋 Loaded {len(jobs)} jobs from {args.config}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Jobs to execute:")
        for job in jobs:
            print(f"\n  • {job['name']}:")
            print(f"    Period: {job['period']['start']} to {job['period']['end']}")
            print(f"    Top N: {job['top_n']}")
            print(f"    Limit pairs: {job.get('limit_pairs', 'unlimited')}")
        return 0
    
    # Run jobs sequentially
    failed_jobs = []
    successful_jobs = []
    
    for i, job in enumerate(jobs, 1):
        print(f"\n🔄 Job {i}/{len(jobs)}: {job['name']}")
        
        result = run_job(job, data_config)
        if result != 0:
            failed_jobs.append(job['name'])
            print(f"⚠️ Continuing despite failure in {job['name']}")
        else:
            successful_jobs.append(job['name'])
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total jobs: {len(jobs)}")
    print(f"✅ Successful: {len(successful_jobs)}")
    print(f"❌ Failed: {len(failed_jobs)}")
    
    if successful_jobs:
        print(f"\n✅ Successful jobs:")
        for name in successful_jobs:
            print(f"  • {name}")
            print(f"    → artifacts/universe/{name}/")
    
    if failed_jobs:
        print(f"\n❌ Failed jobs:")
        for name in failed_jobs:
            print(f"  • {name}")
    
    # Create consolidated report
    report_path = Path('artifacts/universe/JOBS_SUMMARY.md')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(f"""# Universe Selection Jobs Summary

## Execution Details
- **Config**: {args.config}
- **Executed**: {datetime.now(timezone.utc).isoformat()}
- **Total Jobs**: {len(jobs)}
- **Successful**: {len(successful_jobs)}
- **Failed**: {len(failed_jobs)}

## Jobs Results

""")
        
        for job in jobs:
            status = "✅ SUCCESS" if job['name'] in successful_jobs else "❌ FAILED"
            f.write(f"""### {job['name']} - {status}
- Period: {job['period']['start']} to {job['period']['end']}
- Top N: {job['top_n']}
- Output: `artifacts/universe/{job['name']}/`

""")
    
    print(f"\n📄 Summary saved to {report_path}")
    
    return 1 if failed_jobs else 0


if __name__ == '__main__':
    sys.exit(main())