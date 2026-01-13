import os
import glob
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data'))
CLEAN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data_clean'))

os.makedirs(CLEAN_DIR, exist_ok=True)

TARGET_FREQ = '15min'


def audit_and_clean_parquet_files(data_dir: str = DATA_DIR, output_dir: str = CLEAN_DIR, target_freq: str = TARGET_FREQ) -> None:
    """
    Анализирует структуру папок и parquet-файлов, удаляет дубли, выравнивает по target_freq,
    сохраняет очищенные данные в output_dir.
    """
    all_files = glob.glob(os.path.join(data_dir, '**', '*.parquet'), recursive=True)
    print(f'Найдено файлов: {len(all_files)}')
    for file_path in tqdm(all_files, desc='Анализ и чистка parquet'):
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f'[ERROR] {file_path}: {e}')
            continue
        # Проверка наличия timestamp
        if 'timestamp' not in df.columns:
            print(f'[SKIP] Нет timestamp: {file_path}')
            continue
        # Переводим timestamp в datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.sort_values('timestamp').drop_duplicates('timestamp')
        df = df.set_index('timestamp')
        # Выравнивание по target_freq
        idx = pd.date_range(df.index.min(), df.index.max(), freq=target_freq)

        # Логируем количество строк до и после реиндексации, чтобы понять,
        # сколько новых значений (с потенциальными NaN) будет добавлено
        original_rows = len(df)
        projected_rows = len(idx)
        print(
            f"INFO: Reindexing data. Original rows: {original_rows}, "
            f"Projected rows: {projected_rows}. Added rows: {projected_rows - original_rows}"
        )

        df = df.reindex(idx)
        # Логируем диагностику
        n_dups = df.index.duplicated().sum()
        n_missing = df.isna().all(axis=1).sum()
        print(f'{os.path.basename(file_path)}: {len(df)} точек, дубликатов: {n_dups}, пропусков: {n_missing}')
        # Сохраняем очищенный файл
        rel_path = os.path.relpath(file_path, data_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path)

if __name__ == '__main__':
    audit_and_clean_parquet_files()
