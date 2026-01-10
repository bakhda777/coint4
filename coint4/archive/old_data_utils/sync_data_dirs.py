import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from pandas import DataFrame

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    # Fallback if tqdm is not available – use a dummy iterator
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


def collect_parquet_files(root: Path) -> List[Path]:
    """Recursively collect all parquet files under *root*.

    Parameters
    ----------
    root : Path
        Directory to scan.

    Returns
    -------
    List[Path]
        List of absolute paths to parquet files.
    """
    return [p for p in root.rglob("*.parquet") if p.is_file()]


def sync_file(src_file: Path, src_root: Path, dst_root: Path) -> Tuple[int, int]:
    """Synchronise a single parquet *src_file* into *dst_root*.

    The function guarantees that the resulting parquet in *dst_root*
    contains **all** rows from both the destination (if it existed)
    and the source, **without duplicates**.

    Parameters
    ----------
    src_file : Path
        Absolute path to the source parquet file.
    src_root : Path
        Root directory of the *data* folder. Used to compute the relative
        path.
    dst_root : Path
        Root directory of the *data_optimized* folder.

    Returns
    -------
    Tuple[int, int]
        (added_rows, total_rows_after_merge)
    """
    rel_path: Path = src_file.relative_to(src_root)
    dst_file: Path = dst_root / rel_path

    # Ensure destination directory exists.
    dst_file.parent.mkdir(parents=True, exist_ok=True)

    if not dst_file.exists():
        # No file yet – simply copy.
        shutil.copy2(src_file, dst_file)
        df_src: DataFrame = pd.read_parquet(src_file)
        return len(df_src), len(df_src)

    # Destination exists – merge and deduplicate.
    df_src: DataFrame = pd.read_parquet(src_file)
    df_dst: DataFrame = pd.read_parquet(dst_file)

    combined: DataFrame = (
        pd.concat([df_dst, df_src], ignore_index=True)
        .drop_duplicates()
        .sort_values(df_dst.columns.tolist())  # consistent ordering
        .reset_index(drop=True)
    )

    added_rows: int = len(combined) - len(df_dst)

    if added_rows:
        combined.to_parquet(dst_file, index=False)

    return added_rows, len(combined)


def sync_directories(data_dir: Path, optimized_dir: Path) -> None:
    """Main synchronisation routine.

    Traverse *data_dir* and ensure that every parquet file is represented
    in *optimized_dir*. Missing files are copied, and overlapping files
    are merged with duplicate removal.
    """
    parquet_files: List[Path] = collect_parquet_files(data_dir)

    total_added = 0
    for src in tqdm(parquet_files, desc="Synchronising"):  # type: ignore[arg-type]
        added, _ = sync_file(src, data_dir, optimized_dir)
        total_added += added

    print(f"\nСинхронизация завершена. Добавлено строк: {total_added}.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Сравнивает директории data и data_optimized, добавляя отсутствующие "
            "данные без повторов."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Путь к исходной директории data.",
    )
    parser.add_argument(
        "--optimized-dir",
        type=Path,
        default=Path("data_optimized"),
        help="Путь к директории data_optimized.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Директория {args.data_dir} не найдена.")
    if not args.optimized_dir.exists():
        raise FileNotFoundError(f"Директория {args.optimized_dir} не найдена.")

    sync_directories(args.data_dir, args.optimized_dir)


if __name__ == "__main__":
    main()
