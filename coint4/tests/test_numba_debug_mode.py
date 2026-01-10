"""
Тесты для debug-режима numba движка.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from src.coint2.core.numba_parity_v3 import (
    compute_positions_parity,
    compute_positions_parity_debug
)
from src.coint2.utils.traces import save_trace, load_trace, compare_traces


def create_synthetic_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Создать синтетическую коинтегрированную пару."""
    np.random.seed(seed)
    
    # Генерируем x
    x = 100 + np.cumsum(np.random.randn(n) * 0.1)
    
    # y коинтегрирован с x
    beta = 1.2
    ar_component = np.zeros(n)
    for i in range(1, n):
        ar_component[i] = 0.7 * ar_component[i-1] + np.random.randn() * 0.3
    
    y = beta * x + ar_component + np.random.randn(n) * 0.5
    
    dates = pd.date_range('2024-01-01', periods=n, freq='15min')
    return pd.DataFrame({'y': y, 'x': x}, index=dates)


class TestNumbaDebugMode:
    """Тесты debug-режима numba движка."""
    
    def test_normal_mode_signature(self):
        """Тест что обычный режим не меняет сигнатуру."""
        df = create_synthetic_data(500)
        y = df['y'].to_numpy()
        x = df['x'].to_numpy()
        
        # Обычная функция должна возвращать 5 массивов
        result = compute_positions_parity(
            y, x,
            rolling_window=60,
            z_enter=2.0,
            z_exit=0.5,
            max_holding_period=100,
            commission=0.0004,
            slippage=0.0005
        )
        
        assert len(result) == 5, "Обычная функция должна возвращать 5 массивов"
        positions, trades, pnl_series, z_scores, spreads = result
        
        # Проверяем типы и размеры
        assert isinstance(positions, np.ndarray)
        assert isinstance(trades, np.ndarray)
        assert isinstance(pnl_series, np.ndarray)
        assert isinstance(z_scores, np.ndarray)
        assert isinstance(spreads, np.ndarray)
        
        assert len(positions) == len(y)
        assert len(trades) == len(y)
        assert len(pnl_series) == len(y)
        assert len(z_scores) == len(y)
        assert len(spreads) == len(y)
    
    def test_debug_mode_extended_output(self):
        """Тест что debug-режим возвращает расширенный набор данных."""
        df = create_synthetic_data(500)
        y = df['y'].to_numpy()
        x = df['x'].to_numpy()
        
        # Debug функция должна возвращать 11 массивов
        result = compute_positions_parity_debug(
            y, x,
            rolling_window=60,
            z_enter=2.0,
            z_exit=0.5,
            max_holding_period=100,
            commission=0.0004,
            slippage=0.0005
        )
        
        assert len(result) == 11, "Debug функция должна возвращать 11 массивов"
        
        (positions, trades, pnl_series, z_scores, spreads,
         entries_idx, exits_idx, mu, sigma, beta, alpha) = result
        
        # Проверяем типы
        assert isinstance(entries_idx, np.ndarray)
        assert isinstance(exits_idx, np.ndarray)
        assert isinstance(mu, np.ndarray)
        assert isinstance(sigma, np.ndarray)
        assert isinstance(beta, np.ndarray)
        assert isinstance(alpha, np.ndarray)
        
        # Проверяем размеры
        n = len(y)
        assert len(entries_idx) == n
        assert len(exits_idx) == n
        assert len(mu) == n
        assert len(sigma) == n
        assert len(beta) == n
        assert len(alpha) == n
        
        # Проверяем что entries_idx и exits_idx булевы
        assert entries_idx.dtype == np.bool_
        assert exits_idx.dtype == np.bool_
    
    def test_debug_mode_generates_signals(self):
        """Тест что debug-режим генерирует сигналы входа/выхода."""
        df = create_synthetic_data(1000)
        y = df['y'].to_numpy()
        x = df['x'].to_numpy()
        
        result = compute_positions_parity_debug(
            y, x,
            rolling_window=60,
            z_enter=2.0,
            z_exit=0.5,
            max_holding_period=100,
            commission=0.0004,
            slippage=0.0005
        )
        
        entries_idx = result[5]
        exits_idx = result[6]
        
        # Должен быть хотя бы один вход и выход
        assert np.sum(entries_idx) > 0, "Должен быть хотя бы один вход"
        assert np.sum(exits_idx) > 0, "Должен быть хотя бы один выход"
        
        # Количество входов и выходов должно быть примерно равным
        num_entries = np.sum(entries_idx)
        num_exits = np.sum(exits_idx)
        assert abs(num_entries - num_exits) <= 1, "Количество входов и выходов должно быть сбалансировано"
    
    def test_debug_normal_consistency(self):
        """Тест что основные результаты одинаковы в обычном и debug режимах."""
        df = create_synthetic_data(500)
        y = df['y'].to_numpy()
        x = df['x'].to_numpy()
        
        params = {
            'rolling_window': 60,
            'z_enter': 2.0,
            'z_exit': 0.5,
            'max_holding_period': 100,
            'commission': 0.0004,
            'slippage': 0.0005
        }
        
        # Обычный режим
        normal_result = compute_positions_parity(y, x, **params)
        positions_normal, trades_normal, pnl_normal, z_normal, spreads_normal = normal_result
        
        # Debug режим
        debug_result = compute_positions_parity_debug(y, x, **params)
        positions_debug, trades_debug, pnl_debug, z_debug, spreads_debug = debug_result[:5]
        
        # Основные массивы должны быть идентичны
        np.testing.assert_array_equal(positions_normal, positions_debug, "Позиции должны совпадать")
        np.testing.assert_array_equal(trades_normal, trades_debug, "Сделки должны совпадать")
        np.testing.assert_array_almost_equal(pnl_normal, pnl_debug, decimal=10, err_msg="PnL должен совпадать")
        
        # Z-scores могут иметь небольшие численные различия
        valid_mask = ~(np.isnan(z_normal) | np.isnan(z_debug))
        if np.sum(valid_mask) > 0:
            np.testing.assert_array_almost_equal(
                z_normal[valid_mask], 
                z_debug[valid_mask], 
                decimal=10,
                err_msg="Z-scores должны совпадать"
            )
    
    def test_trace_saving_loading(self):
        """Тест сохранения и загрузки трейсов."""
        df = create_synthetic_data(200)
        y = df['y'].to_numpy()
        x = df['x'].to_numpy()
        
        # Получаем debug данные
        result = compute_positions_parity_debug(
            y, x,
            rolling_window=60,
            z_enter=2.0,
            z_exit=0.5,
            max_holding_period=100,
            commission=0.0004,
            slippage=0.0005
        )
        
        (positions, trades, pnl_series, z_scores, spreads,
         entries_idx, exits_idx, mu, sigma, beta, alpha) = result
        
        # Сохраняем трейс
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "test_trace.csv"
            
            meta = {
                'engine': 'numba',
                'roll': 60,
                'z_enter': 2.0,
                'z_exit': 0.5,
                'pair': 'TEST',
                'timeframe': 'M15'
            }
            
            saved_path = save_trace(
                df.index,
                z_scores,
                entries_idx,
                exits_idx,
                positions,
                pnl_series,
                trace_path,
                meta,
                spreads=spreads,
                mu=mu,
                sigma=sigma,
                beta=beta,
                alpha=alpha
            )
            
            assert saved_path.exists(), "Файл трейса должен быть создан"
            
            # Загружаем трейс
            loaded_df, loaded_meta = load_trace(saved_path)
            
            # Проверяем метаданные
            assert loaded_meta['engine'] == 'numba'
            assert loaded_meta['roll'] == 60
            assert loaded_meta['z_enter'] == 2.0
            
            # Проверяем данные
            assert len(loaded_df) == len(df)
            assert 'z_score' in loaded_df.columns
            assert 'entry' in loaded_df.columns
            assert 'exit' in loaded_df.columns
            assert 'position' in loaded_df.columns
            assert 'spread' in loaded_df.columns
            assert 'mu' in loaded_df.columns
            assert 'sigma' in loaded_df.columns
            
            # Проверяем что данные совпадают
            np.testing.assert_array_almost_equal(
                loaded_df['position'].values,
                positions,
                decimal=10
            )
    
    def test_compare_traces_identical(self):
        """Тест сравнения идентичных трейсов."""
        df = create_synthetic_data(200)
        y = df['y'].to_numpy()
        x = df['x'].to_numpy()
        
        params = {
            'rolling_window': 60,
            'z_enter': 2.0,
            'z_exit': 0.5,
            'max_holding_period': 100,
            'commission': 0.0004,
            'slippage': 0.0005
        }
        
        # Получаем debug данные
        result = compute_positions_parity_debug(y, x, **params)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Сохраняем два идентичных трейса
            trace1_path = Path(tmpdir) / "trace1.csv"
            trace2_path = Path(tmpdir) / "trace2.csv"
            
            meta1 = {'engine': 'numba1', 'roll': 60}
            meta2 = {'engine': 'numba2', 'roll': 60}
            
            for path, meta in [(trace1_path, meta1), (trace2_path, meta2)]:
                save_trace(
                    df.index,
                    result[3],  # z_scores
                    result[5],  # entries_idx
                    result[6],  # exits_idx
                    result[0],  # positions
                    result[2],  # pnl_series
                    path,
                    meta
                )
            
            # Сравниваем трейсы
            comparison = compare_traces(trace1_path, trace2_path)
            
            # Идентичные трейсы должны иметь 100% совпадение
            assert comparison['position_match_pct'] == 100.0
            assert comparison['entry_match_pct'] == 100.0
            assert comparison['exit_match_pct'] == 100.0
            assert comparison['z_score_correlation'] > 0.999
            assert comparison['pnl_diff'] < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])