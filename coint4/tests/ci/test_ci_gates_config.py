"""Tests for CI gates configuration and source handling."""

import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.ci_gates import CIGateChecker


class TestCIGatesConfig:
    """Test CI gates configuration handling."""
    
    def test_default_config(self):
        """Test loading with default configuration."""
        checker = CIGateChecker(config_path='nonexistent.yaml', verbose=False)
        assert checker.config is not None
        assert 'performance' in checker.config
        assert 'thresholds' in checker.config
    
    def test_custom_config_loading(self):
        """Test loading custom configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'performance': {
                    'source': 'wfa',
                    'wfa': {
                        'path': 'test.csv',
                        'sharpe_col': 'sharpe_ratio',
                        'min_trades': 5
                    }
                },
                'thresholds': {
                    'min_sharpe': 1.0,
                    'max_drawdown_pct': 20.0
                }
            }
            yaml.dump(config, f)
            f.flush()
            
            checker = CIGateChecker(config_path=f.name, verbose=False)
            assert checker.config['performance']['source'] == 'wfa'
            assert checker.config['thresholds']['min_sharpe'] == 1.0
            
            Path(f.name).unlink()
    
    def test_missing_file_fail_explicit(self):
        """Test fail_explicit behavior on missing file."""
        checker = CIGateChecker(verbose=False)
        checker.config['fallbacks'] = {'on_missing_file': 'fail_explicit'}
        checker.config['performance'] = {
            'source': 'wfa',
            'wfa': {'path': 'nonexistent.csv'}
        }
        
        passed, metrics = checker.check_performance_metrics()
        assert not passed
        assert 'error' in metrics
        assert 'not found' in metrics['error']
    
    def test_empty_csv_handling(self):
        """Test handling of empty CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write empty CSV with headers only
            f.write("sharpe,trades,pnl\n")
            f.flush()
            
            checker = CIGateChecker(verbose=False)
            checker.config['performance'] = {
                'source': 'wfa',
                'wfa': {
                    'path': f.name,
                    'sharpe_col': 'sharpe',
                    'trades_col': 'trades',
                    'min_trades': 10
                }
            }
            checker.config['fallbacks'] = {'on_insufficient_trades': 'fail_explicit'}
            
            passed, metrics = checker.check_performance_metrics()
            assert not passed
            assert 'error' in metrics or metrics.get('total_trades', 0) == 0
            
            Path(f.name).unlink()
    
    def test_valid_csv_processing(self):
        """Test processing of valid CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write valid CSV data
            f.write("sharpe,trades,pnl\n")
            f.write("1.5,20,1000\n")
            f.write("1.8,25,1200\n")
            f.write("2.0,30,1500\n")
            f.flush()
            
            checker = CIGateChecker(verbose=False)
            checker.config['performance'] = {
                'source': 'wfa',
                'wfa': {
                    'path': f.name,
                    'sharpe_col': 'sharpe',
                    'trades_col': 'trades',
                    'pnl_col': 'pnl',
                    'min_trades': 10,
                    'aggregation': 'mean'
                }
            }
            checker.config['thresholds'] = {
                'min_sharpe': 1.0,
                'max_drawdown_pct': 50.0,
                'min_trades': 10
            }
            
            passed, metrics = checker.check_performance_metrics()
            assert passed
            assert metrics['sharpe_ratio'] == pytest.approx(1.77, rel=0.01)
            assert metrics['total_trades'] == 75
            
            Path(f.name).unlink()
    
    def test_insufficient_trades(self):
        """Test handling of insufficient trades."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write CSV with very few trades
            f.write("sharpe,trades,pnl\n")
            f.write("2.5,2,100\n")
            f.write("3.0,3,150\n")
            f.flush()
            
            checker = CIGateChecker(verbose=False)
            checker.config['performance'] = {
                'source': 'wfa',
                'wfa': {
                    'path': f.name,
                    'sharpe_col': 'sharpe',
                    'trades_col': 'trades',
                    'min_trades': 20
                }
            }
            checker.config['fallbacks'] = {'on_insufficient_trades': 'fail_explicit'}
            
            passed, metrics = checker.check_performance_metrics()
            assert not passed
            assert 'Insufficient trades' in str(metrics.get('error', ''))
            
            Path(f.name).unlink()
    
    def test_aggregation_methods(self):
        """Test different aggregation methods."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write CSV with varied Sharpe values
            f.write("sharpe,trades,pnl\n")
            f.write("1.0,20,1000\n")
            f.write("2.0,25,1200\n")
            f.write("3.0,30,1500\n")
            f.flush()
            
            # Test mean aggregation
            checker = CIGateChecker(verbose=False)
            checker.config['performance'] = {
                'source': 'wfa',
                'wfa': {
                    'path': f.name,
                    'sharpe_col': 'sharpe',
                    'trades_col': 'trades',
                    'aggregation': 'mean'
                }
            }
            
            _, metrics_mean = checker.check_performance_metrics()
            assert metrics_mean['sharpe_ratio'] == 2.0
            
            # Test median aggregation
            checker.config['performance']['wfa']['aggregation'] = 'median'
            _, metrics_median = checker.check_performance_metrics()
            assert metrics_median['sharpe_ratio'] == 2.0
            
            Path(f.name).unlink()
    
    def test_verbose_output(self, capsys):
        """Test verbose output generation."""
        checker = CIGateChecker(verbose=True)
        
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("sharpe,trades\n1.5,50\n")
            f.flush()
            
            checker.config['performance']['wfa']['path'] = f.name
            checker.check_performance_metrics()
            
            captured = capsys.readouterr()
            assert "Reading performance from source" in captured.out
            assert "Loaded" in captured.out
            
            Path(f.name).unlink()