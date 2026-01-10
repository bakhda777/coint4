# Large file storage

This repo keeps heavy data and generated outputs out of Git to avoid GitHub size limits and huge history.

## What stays out of Git
- datasets (parquet/csv)
- run outputs (results, logs, optuna studies)
- archives (.zip, .tar.gz)

These live under `data/`, `data_downloaded/`, `outputs/`, `results/`, `logs/`, `artifacts/`, and `archive/`.

## Recommended approach
1. Use external storage for full datasets (S3/Drive/NAS).
2. Keep a small sample for tests in a tracked folder such as `data_sample/` (optional).
3. Use Git LFS only for large files that must be versioned.

## Git LFS quickstart
Install Git LFS and run:

```bash
git lfs install
git lfs track "*.parquet" "*.db" "*.zip"
git add .gitattributes
```

After tracking, commit and push as usual.
