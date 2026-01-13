import os
import glob
import pandas as pd
from tqdm import tqdm

CLEAN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data_clean'))
FINAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data_final'))

os.makedirs(FINAL_DIR, exist_ok=True)

def save_day_partition(df: pd.DataFrame, symbol: str, date: pd.Timestamp, out_base: str):
    year = date.year
    month = date.month
    day = date.day
    out_dir = os.path.join(out_base, f"year={year}", f"month={month:02d}", f"day={day:02d}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{date.strftime('%Y-%m-%d')}.parquet")
    df.to_parquet(out_path)


def partition_clean_data_to_hive(clean_dir: str = CLEAN_DIR, final_dir: str = FINAL_DIR):
    all_files = glob.glob(os.path.join(clean_dir, '**', '*.parquet'), recursive=True)
    print(f'Найдено очищенных файлов: {len(all_files)}')
    for file_path in tqdm(all_files, desc='Партиционирование по дням'):
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f'[ERROR] {file_path}: {e}')
            continue
        if df.empty:
            continue
        # Восстанавливаем symbol (можно доработать, если структура сложнее)
        symbol = os.path.basename(os.path.dirname(file_path))
        # Удаляем дубли по индексу
        df = df[~df.index.duplicated(keep='first')]
        # Группируем по дням
        for date, day_df in df.groupby(df.index.date):
            if day_df.empty:
                continue
            save_day_partition(day_df, symbol, pd.Timestamp(date), final_dir)
            print(f'{symbol} {date}: {len(day_df)} точек, дубликатов: {day_df.index.duplicated().sum()}')

if __name__ == '__main__':
    partition_clean_data_to_hive()
