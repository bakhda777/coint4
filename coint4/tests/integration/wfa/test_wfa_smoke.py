"""
Smoke тесты для системы Walk-Forward Analysis.

Проверяет основную функциональность WFA:
- Загрузка конфигурации
- Создание временных фолдов
- Интеграция с Optuna
- Сохранение результатов

ВАЖНО: Использует минимальные datasets для быстрого выполнения
"""

import pytest
import tempfile
import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

# Импорты из проекта
import sys
sys.path.append(str(Path(__file__).parents[3] / "src"))

pytest.importorskip("scripts.run_walk_forward", reason="Legacy WFA runner not available.")
from scripts.run_walk_forward import WalkForwardAnalyzer
from coint2.utils.config import AppConfig


@pytest.fixture
def temp_wfa_config():
    """Создать временную WFA конфигурацию для тестов."""
    config = {
        "base_config": "../configs/main_2024.yaml",
        "results_dir": "test_artifacts/wfa",
        "walk_forward": {
            "enabled": True,
            "start_date": "2024-01-01",
            "end_date": "2024-02-29", 
            "training_period_days": 30,  # Сокращено для тестов
            "testing_period_days": 15,   # Сокращено для тестов
            "step_size_days": 15,
            "gap_minutes": 15,
            "max_steps": 3,  # Максимум 3 фолда для smoke теста
            "min_training_samples": 100
        },
        "optuna": {
            "n_trials": 3,  # Минимум для smoke теста
            "study_name": "test_wfa",
            "storage": "sqlite:///test_wfa_studies.db",
            "search_space": {
                "zscore_threshold": {"type": "uniform", "low": 1.5, "high": 2.5},
                "zscore_exit": {"type": "uniform", "low": 0.2, "high": 0.8}
            }
        },
        "success_criteria": {
            "min_sharpe_ratio": 0.5,    # Ослаблено для тестов
            "min_trade_count": 5,       # Ослаблено для тестов  
            "max_drawdown_pct": 50.0    # Ослаблено для тестов
        },
        "traces": {
            "enabled": True,
            "save_path": "test_artifacts/wfa/traces"
        },
        "reporting": {
            "generate_fold_reports": True,
            "generate_stability_analysis": True
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        return Path(f.name)


@pytest.fixture
def mock_base_config():
    """Замоканная базовая конфигурация."""
    config = MagicMock(spec=AppConfig)
    config.walk_forward = MagicMock()
    config.backtest = MagicMock()
    config.backtest.zscore_threshold = 2.0
    config.backtest.zscore_exit = 0.5
    config.backtest.rolling_window = 30
    return config


@pytest.fixture  
def sample_market_data():
    """Создать образец рыночных данных для тестов."""
    dates = pd.date_range("2024-01-01", "2024-02-29", freq="15min")
    n = len(dates)
    
    # Простые коинтегрированные ряды
    np.random.seed(42)
    
    # BTC цена с трендом и случайными движениями
    btc_base = 40000
    btc_trend = np.linspace(0, 5000, n)
    btc_noise = np.cumsum(np.random.normal(0, 100, n))
    btc_price = btc_base + btc_trend + btc_noise
    
    # ETH цена коинтегрирована с BTC
    eth_base = 2500
    beta = 0.06  # ETH/BTC отношение
    eth_price = eth_base + beta * btc_price + np.cumsum(np.random.normal(0, 20, n))
    
    data = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["BTC"] * n + ["ETH"] * n,
        "close": np.concatenate([btc_price, eth_price])
    })
    
    return data


class TestWalkForwardAnalyzer:
    """Тесты для класса WalkForwardAnalyzer."""

    @pytest.mark.smoke
    def test_analyzer_initialization(self, temp_wfa_config, mock_base_config):
        """Тест инициализации анализатора."""
        
        with patch('scripts.run_walk_forward.WalkForwardAnalyzer._load_base_config', 
                  return_value=mock_base_config):
            analyzer = WalkForwardAnalyzer(str(temp_wfa_config))
            
            assert analyzer.config_path == temp_wfa_config
            assert analyzer.wfa_config is not None
            assert analyzer.results_dir.name == "wfa"
            assert analyzer.traces_dir.name == "traces"

    @pytest.mark.smoke
    def test_time_folds_creation(self, temp_wfa_config, mock_base_config):
        """Тест создания временных фолдов."""
        
        with patch('scripts.run_walk_forward.WalkForwardAnalyzer._load_base_config', 
                  return_value=mock_base_config):
            analyzer = WalkForwardAnalyzer(str(temp_wfa_config))
            folds = analyzer._create_time_folds()
            
            # Проверить количество фолдов
            assert len(folds) >= 3, f"Ожидается минимум 3 фолда, получено {len(folds)}"
            
            # Проверить структуру фолдов
            for fold in folds:
                assert "fold_id" in fold
                assert "train_start" in fold
                assert "train_end" in fold
                assert "test_start" in fold
                assert "test_end" in fold
                assert "gap_minutes" in fold
                
                # Проверить gap между train и test
                gap = (fold["test_start"] - fold["train_end"]).total_seconds() / 60
                assert gap == 15, f"Gap должен быть 15 минут, получено {gap}"
                
                # Проверить длительности периодов
                train_days = (fold["train_end"] - fold["train_start"]).days
                test_days = (fold["test_end"] - fold["test_start"]).days
                
                assert train_days >= 25, f"Train период слишком короткий: {train_days} дней"
                assert test_days >= 10, f"Test период слишком короткий: {test_days} дней"

    @pytest.mark.smoke
    def test_fold_config_creation(self, temp_wfa_config, mock_base_config):
        """Тест создания конфигурации для фолда."""
        
        with patch('scripts.run_walk_forward.WalkForwardAnalyzer._load_base_config', 
                  return_value=mock_base_config):
            analyzer = WalkForwardAnalyzer(str(temp_wfa_config))
            
            # Создать тестовый фолд
            fold = {
                "fold_id": 1,
                "train_start": datetime(2024, 1, 1),
                "train_end": datetime(2024, 1, 30),
                "test_start": datetime(2024, 1, 31),
                "test_end": datetime(2024, 2, 15)
            }
            
            with patch.object(analyzer, '_load_base_config', return_value=mock_base_config):
                fold_config = analyzer._create_fold_config(fold)
                
                # Проверить, что конфигурация обновлена
                assert fold_config is not None

    @pytest.mark.smoke  
    def test_success_criteria_validation(self, temp_wfa_config, mock_base_config):
        """Тест валидации критериев успеха."""
        
        with patch('scripts.run_walk_forward.WalkForwardAnalyzer._load_base_config', 
                  return_value=mock_base_config):
            analyzer = WalkForwardAnalyzer(str(temp_wfa_config))
            
            # Тест успешного результата
            good_result = {
                "best_sharpe": 1.2,
                "trade_count": 25,
                "max_drawdown": 15.0
            }
            
            validation = analyzer._validate_fold_results(good_result)
            assert validation["sharpe_ok"] == True
            assert validation["trades_ok"] == True
            assert validation["drawdown_ok"] == True
            assert validation["overall_success"] == True
            
            # Тест неуспешного результата
            bad_result = {
                "best_sharpe": 0.2,  # Слишком низкий
                "trade_count": 2,    # Слишком мало сделок
                "max_drawdown": 60.0 # Слишком большая просадка
            }
            
            validation = analyzer._validate_fold_results(bad_result)
            assert validation["overall_success"] == False

    @pytest.mark.smoke
    def test_detailed_metrics_calculation(self, temp_wfa_config, mock_base_config):
        """Тест расчета детальных метрик."""
        
        with patch('scripts.run_walk_forward.WalkForwardAnalyzer._load_base_config', 
                  return_value=mock_base_config):
            analyzer = WalkForwardAnalyzer(str(temp_wfa_config))
            
            # Создать тестовые результаты
            test_results = {
                "pnl": [0, 100, 50, 200, 150, 300],  # Растущий P&L
                "trades": [
                    {"pnl": 100, "entry_time": "2024-01-01 10:00"},
                    {"pnl": -50, "entry_time": "2024-01-01 11:00"},
                    {"pnl": 150, "entry_time": "2024-01-01 12:00"}
                ]
            }
            
            test_data = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=6, freq="1h")})
            
            metrics = analyzer._calculate_detailed_metrics(test_results, test_data)
            
            # Проверить базовые метрики
            assert "sharpe_ratio" in metrics
            assert "total_return" in metrics
            assert "trade_count" in metrics
            assert "max_drawdown" in metrics
            assert "win_rate" in metrics
            assert "avg_trade_return" in metrics
            
            assert metrics["trade_count"] == 3
            assert metrics["total_return"] == 300  # Конечный P&L
            assert 0 <= metrics["win_rate"] <= 100  # Win rate в процентах

    @pytest.mark.smoke
    def test_summary_generation(self, temp_wfa_config, mock_base_config):
        """Тест генерации сводки результатов."""
        
        with patch('scripts.run_walk_forward.WalkForwardAnalyzer._load_base_config', 
                  return_value=mock_base_config):
            analyzer = WalkForwardAnalyzer(str(temp_wfa_config))
            
            # Создать тестовые результаты фолдов
            fold_results = [
                {
                    "fold_id": 1,
                    "sharpe_ratio": 1.2,
                    "trade_count": 25,
                    "max_drawdown": 12.0,
                    "validation": {"overall_success": True}
                },
                {
                    "fold_id": 2, 
                    "sharpe_ratio": 0.8,
                    "trade_count": 15,
                    "max_drawdown": 18.0,
                    "validation": {"overall_success": False}
                },
                {
                    "fold_id": 3,
                    "sharpe_ratio": 1.5,
                    "trade_count": 30,
                    "max_drawdown": 8.0,
                    "validation": {"overall_success": True}
                }
            ]
            
            optimization_results = []
            
            summary = analyzer._generate_summary(fold_results, optimization_results)
            
            # Проверить структуру сводки
            assert "total_folds" in summary
            assert "successful_folds" in summary
            assert "success_rate" in summary
            assert "sharpe_ratio" in summary
            assert "trade_count" in summary
            assert "max_drawdown" in summary
            
            assert summary["total_folds"] == 3
            assert summary["successful_folds"] == 2
            assert summary["success_rate"] == pytest.approx(66.67, abs=0.1)

    @pytest.mark.smoke
    @patch('scripts.run_walk_forward.DataHandler')
    @patch('scripts.run_walk_forward.ReferenceEngine')
    @patch('scripts.run_walk_forward.FastWalkForwardObjective')
    @patch('optuna.create_study')
    def test_full_pipeline_mock(self, mock_study, mock_objective, mock_engine, mock_handler,
                               temp_wfa_config, mock_base_config):
        """Интеграционный smoke тест всего пайплайна с моками."""
        
        # Настроить моки
        mock_study_instance = MagicMock()
        mock_study_instance.best_trial = MagicMock()
        mock_study_instance.best_params = {"zscore_threshold": 2.0, "zscore_exit": 0.5}
        mock_study_instance.best_value = 1.2
        mock_study_instance.trials = [MagicMock() for _ in range(3)]
        mock_study.return_value = mock_study_instance
        
        mock_engine_instance = MagicMock()
        mock_engine_instance.backtest_pair.return_value = {
            "pnl": [0, 100, 200, 150, 300],
            "trades": [{"pnl": 100}, {"pnl": 50}, {"pnl": 150}]
        }
        mock_engine_instance.debug_data = {
            "z": np.array([0, 1, 2, 1, 0]),
            "entries_idx": np.array([False, True, False, True, False]),
            "exits_idx": np.array([False, False, True, False, True]),
            "positions": np.array([0, 1, 0, 1, 0]),
            "pnl": np.array([0, 100, 200, 150, 300])
        }
        mock_engine.return_value = mock_engine_instance
        
        mock_handler_instance = MagicMock()
        mock_handler_instance.load_pair_data.return_value = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="15min"),
            "close_BTC": np.random.randn(100) + 40000,
            "close_ETH": np.random.randn(100) + 2500
        })
        mock_handler.return_value = mock_handler_instance
        
        # Запустить анализ
        with patch('scripts.run_walk_forward.WalkForwardAnalyzer._load_base_config', 
                  return_value=mock_base_config):
            
            analyzer = WalkForwardAnalyzer(str(temp_wfa_config))
            
            # Создать 3 фолда для быстрого smoke теста  
            with patch.object(analyzer, '_create_time_folds') as mock_folds:
                mock_folds.return_value = [
                    {
                        "fold_id": 1,
                        "train_start": datetime(2024, 1, 1),
                        "train_end": datetime(2024, 1, 30),
                        "test_start": datetime(2024, 1, 31),
                        "test_end": datetime(2024, 2, 15),
                        "gap_minutes": 15
                    },
                    {
                        "fold_id": 2,
                        "train_start": datetime(2024, 2, 1),
                        "train_end": datetime(2024, 2, 28),
                        "test_start": datetime(2024, 3, 1),
                        "test_end": datetime(2024, 3, 15),
                        "gap_minutes": 15
                    },
                    {
                        "fold_id": 3,
                        "train_start": datetime(2024, 3, 1),
                        "train_end": datetime(2024, 3, 30),
                        "test_start": datetime(2024, 3, 31),
                        "test_end": datetime(2024, 4, 15),
                        "gap_minutes": 15
                    }
                ]
                
                summary = analyzer.run_analysis()
                
                # Проверить базовые результаты
                assert summary is not None
                assert "total_folds" in summary
                assert summary["total_folds"] == 3


@pytest.mark.smoke
def test_wfa_script_imports():
    """Smoke тест импортов скрипта WFA."""
    
    # Проверить, что все импорты работают
    from scripts.run_walk_forward import (
        WalkForwardAnalyzer, 
        generate_report,
        main
    )
    
    assert WalkForwardAnalyzer is not None
    assert generate_report is not None
    assert main is not None


@pytest.mark.smoke 
def test_wfa_config_validation():
    """Smoke тест валидации WFA конфигурации."""
    
    # Проверить существование основных файлов
    config_path = Path(__file__).parents[3] / "bench" / "wfa.yaml"
    assert config_path.exists(), f"WFA конфигурация не найдена: {config_path}"
    
    # Проверить структуру конфигурации
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Проверить обязательные секции
    required_sections = ["walk_forward", "optuna", "success_criteria", "traces"]
    for section in required_sections:
        assert section in config, f"Отсутствует секция {section} в конфигурации"
    
    # Проверить критические параметры
    wf = config["walk_forward"]
    assert wf["gap_minutes"] == 15, "Gap должен быть 15 минут"
    assert wf["training_period_days"] >= 60, "Training период должен быть ≥60 дней"
    assert wf["testing_period_days"] >= 30, "Testing период должен быть ≥30 дней"


if __name__ == "__main__":
    # Запуск smoke тестов
    pytest.main([__file__, "-v", "-m", "smoke"])
