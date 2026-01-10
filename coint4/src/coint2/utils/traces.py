"""
Модуль для сохранения и загрузки трейсов торговых движков.
Используется для отладки, сравнения движков и анализа торговых решений.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Optional, Union
import subprocess


def get_git_commit() -> str:
    """Получить хеш последнего коммита."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except:
        return "unknown"


def save_trace(
    df_index: pd.Index,
    z: np.ndarray,
    entries_idx: np.ndarray,
    exits_idx: np.ndarray,
    positions: np.ndarray,
    pnl: np.ndarray,
    out_path: Union[str, Path],
    meta: Dict,
    spreads: Optional[np.ndarray] = None,
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    beta: Optional[np.ndarray] = None,
    alpha: Optional[np.ndarray] = None
) -> Path:
    """
    Сохранить трейс торгового движка в CSV файл.
    
    Args:
        df_index: Индекс временного ряда (даты/время)
        z: Массив z-scores
        entries_idx: Булев массив точек входа
        exits_idx: Булев массив точек выхода
        positions: Массив позиций (-1, 0, 1)
        pnl: Массив PnL
        out_path: Путь для сохранения
        meta: Метаданные (engine, roll, z_enter, z_exit, fees, ts, git)
        spreads: Опционально - массив спредов
        mu: Опционально - массив средних
        sigma: Опционально - массив стандартных отклонений
        beta: Опционально - массив бета коэффициентов
        alpha: Опционально - массив альфа коэффициентов
    
    Returns:
        Path: Путь к сохраненному файлу
    """
    # Создаем директорию если не существует
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Добавляем git commit и timestamp если не указаны
    if 'git' not in meta:
        meta['git'] = get_git_commit()
    if 'ts' not in meta:
        meta['ts'] = datetime.now().isoformat()
    
    # Создаем DataFrame
    data = {
        'timestamp': df_index,
        'z_score': z,
        'entry': entries_idx.astype(int),
        'exit': exits_idx.astype(int),
        'position': positions,
        'pnl': pnl,
        'cum_pnl': np.cumsum(pnl)
    }
    
    # Добавляем опциональные поля
    if spreads is not None:
        data['spread'] = spreads
    if mu is not None:
        data['mu'] = mu
    if sigma is not None:
        data['sigma'] = sigma
    if beta is not None:
        data['beta'] = beta
    if alpha is not None:
        data['alpha'] = alpha
    
    df = pd.DataFrame(data)
    
    # Сохраняем метаданные в первой строке как комментарий
    with open(out_path, 'w') as f:
        f.write(f"# META: {json.dumps(meta)}\n")
        df.to_csv(f, index=False)
    
    return out_path


def load_trace(path: Union[str, Path]) -> tuple:
    """
    Загрузить трейс из файла.
    
    Args:
        path: Путь к файлу трейса
    
    Returns:
        tuple: (df, meta) - DataFrame с данными и словарь метаданных
    """
    path = Path(path)
    
    # Читаем метаданные из первой строки
    with open(path, 'r') as f:
        meta_line = f.readline()
        if meta_line.startswith('# META:'):
            meta = json.loads(meta_line[7:].strip())
        else:
            meta = {}
    
    # Читаем DataFrame, пропуская первую строку с метаданными
    df = pd.read_csv(path, comment='#')
    
    return df, meta


def compare_traces(
    trace1_path: Union[str, Path],
    trace2_path: Union[str, Path],
    tolerance_bars: int = 1
) -> Dict:
    """
    Сравнить два трейса и вычислить метрики паритета.
    
    Args:
        trace1_path: Путь к первому трейсу
        trace2_path: Путь ко второму трейсу
        tolerance_bars: Допустимое расхождение в барах для точек входа/выхода
    
    Returns:
        Dict: Словарь с метриками сравнения
    """
    df1, meta1 = load_trace(trace1_path)
    df2, meta2 = load_trace(trace2_path)
    
    # Убеждаемся что длины совпадают
    min_len = min(len(df1), len(df2))
    df1 = df1.iloc[:min_len]
    df2 = df2.iloc[:min_len]
    
    # Сравнение позиций
    position_match = np.sum(df1['position'] == df2['position']) / min_len * 100
    
    # Сравнение точек входа (с допуском)
    entries1 = np.where(df1['entry'] == 1)[0]
    entries2 = np.where(df2['entry'] == 1)[0]
    
    entry_matches = 0
    for e1 in entries1:
        # Проверяем есть ли вход в trace2 в пределах tolerance_bars
        if np.any(np.abs(entries2 - e1) <= tolerance_bars):
            entry_matches += 1
    
    entry_match_pct = entry_matches / max(len(entries1), 1) * 100
    
    # Сравнение точек выхода
    exits1 = np.where(df1['exit'] == 1)[0]
    exits2 = np.where(df2['exit'] == 1)[0]
    
    exit_matches = 0
    for e1 in exits1:
        if np.any(np.abs(exits2 - e1) <= tolerance_bars):
            exit_matches += 1
    
    exit_match_pct = exit_matches / max(len(exits1), 1) * 100
    
    # Корреляция z-scores
    valid_mask = ~(np.isnan(df1['z_score']) | np.isnan(df2['z_score']))
    if np.sum(valid_mask) > 0:
        z_corr = np.corrcoef(df1['z_score'][valid_mask], df2['z_score'][valid_mask])[0, 1]
    else:
        z_corr = 0
    
    # PnL метрики
    total_pnl1 = df1['cum_pnl'].iloc[-1] if len(df1) > 0 else 0
    total_pnl2 = df2['cum_pnl'].iloc[-1] if len(df2) > 0 else 0
    
    return {
        'engine1': meta1.get('engine', 'unknown'),
        'engine2': meta2.get('engine', 'unknown'),
        'position_match_pct': position_match,
        'entry_match_pct': entry_match_pct,
        'exit_match_pct': exit_match_pct,
        'z_score_correlation': z_corr,
        'total_pnl1': total_pnl1,
        'total_pnl2': total_pnl2,
        'pnl_diff': abs(total_pnl1 - total_pnl2),
        'num_entries1': len(entries1),
        'num_entries2': len(entries2),
        'num_exits1': len(exits1),
        'num_exits2': len(exits2)
    }


def create_trace_index(traces_dir: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Создать индекс всех трейсов в директории.
    
    Args:
        traces_dir: Директория с трейсами
        output_path: Путь для сохранения индекса (опционально)
    
    Returns:
        pd.DataFrame: DataFrame с информацией о всех трейсах
    """
    traces_dir = Path(traces_dir)
    
    if not traces_dir.exists():
        traces_dir.mkdir(parents=True, exist_ok=True)
        return pd.DataFrame()
    
    index_data = []
    
    for trace_file in traces_dir.glob("*.csv"):
        try:
            df, meta = load_trace(trace_file)
            
            # Считаем статистику
            entries = np.sum(df['entry'] == 1)
            exits = np.sum(df['exit'] == 1)
            max_abs_z = np.nanmax(np.abs(df['z_score'])) if 'z_score' in df.columns else 0
            
            index_data.append({
                'file': trace_file.name,
                'path': str(trace_file),
                'engine': meta.get('engine', 'unknown'),
                'pair': meta.get('pair', 'unknown'),
                'timeframe': meta.get('timeframe', 'unknown'),
                'roll': meta.get('roll', 0),
                'z_enter': meta.get('z_enter', 0),
                'z_exit': meta.get('z_exit', 0),
                'n_entries': entries,
                'n_exits': exits,
                'max_abs_z': max_abs_z,
                'total_pnl': df['cum_pnl'].iloc[-1] if len(df) > 0 else 0,
                'timestamp': meta.get('ts', ''),
                'git_commit': meta.get('git', '')
            })
        except Exception as e:
            print(f"Error processing {trace_file}: {e}")
    
    index_df = pd.DataFrame(index_data)
    
    if output_path:
        output_path = Path(output_path)
        index_df.to_csv(output_path, index=False)
    
    return index_df