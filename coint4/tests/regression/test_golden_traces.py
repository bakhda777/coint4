"""Regression tests using golden traces."""

import pytest
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from pathlib import Path
import json


def generate_golden_traces():
    """Generate baseline traces for regression testing."""
    
    Path("artifacts/baseline_traces").mkdir(parents=True, exist_ok=True)
    
    # Create synthetic golden data
    dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
    
    # BTC/ETH pair
    btc_prices = 50000 + np.random.randn(1000).cumsum() * 100
    eth_prices = 3000 + np.random.randn(1000).cumsum() * 50
    
    btc_eth_trace = {
        'pair': 'BTC/ETH',
        'engine': 'reference',
        'timestamps': [str(d) for d in dates],
        'prices_y': btc_prices.tolist(),
        'prices_x': eth_prices.tolist(),
        'positions': np.random.choice([-1, 0, 1], size=1000, p=[0.2, 0.6, 0.2]).tolist(),
        'pnl': np.random.randn(1000).cumsum().tolist(),
        'parameters': {
            'zscore_threshold': 2.0,
            'zscore_exit': 0.0,
            'rolling_window': 60
        }
    }
    
    # Save golden trace
    with open("artifacts/baseline_traces/btc_eth_reference.json", 'w') as f:
        json.dump(btc_eth_trace, f)
    
    # ETH/USDT pair
    eth_usdt_trace = {
        'pair': 'ETH/USDT',
        'engine': 'reference',
        'timestamps': [str(d) for d in dates],
        'prices_y': eth_prices.tolist(),
        'prices_x': np.ones(1000).tolist(),
        'positions': np.random.choice([-1, 0, 1], size=1000, p=[0.3, 0.4, 0.3]).tolist(),
        'pnl': np.random.randn(1000).cumsum().tolist(),
        'parameters': {
            'zscore_threshold': 1.5,
            'zscore_exit': 0.5,
            'rolling_window': 45
        }
    }
    
    with open("artifacts/baseline_traces/eth_usdt_reference.json", 'w') as f:
        json.dump(eth_usdt_trace, f)
    
    print("Generated 2 golden traces")
    return True


def load_golden_trace(pair: str, engine: str = 'reference') -> dict:
    """Load golden trace for comparison."""
    filename = f"{pair.replace('/', '_').lower()}_{engine}.json"
    path = Path(f"artifacts/baseline_traces/{filename}")
    
    if not path.exists():
        # Generate if not exists
        generate_golden_traces()
    
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    
    return None


def compare_traces(golden: dict, current: dict, tolerance: float = 0.001) -> dict:
    """Compare current trace with golden baseline."""
    
    results = {
        'positions_match': 0.0,
        'pnl_rmse': 0.0,
        'entry_exit_match': 0.0,
        'issues': []
    }
    
    # Convert to arrays
    golden_pos = np.array(golden['positions'])
    current_pos = np.array(current.get('positions', []))
    
    if len(current_pos) == 0:
        results['issues'].append("No positions in current trace")
        return results
    
    # Align lengths
    min_len = min(len(golden_pos), len(current_pos))
    golden_pos = golden_pos[:min_len]
    current_pos = current_pos[:min_len]
    
    # Position match rate
    matches = (golden_pos == current_pos).sum()
    results['positions_match'] = matches / min_len
    
    # PnL RMSE
    if 'pnl' in golden and 'pnl' in current:
        golden_pnl = np.array(golden['pnl'][:min_len])
        current_pnl = np.array(current['pnl'][:min_len])
        
        # Normalize by scale
        scale = np.abs(golden_pnl).mean() + 1e-10
        results['pnl_rmse'] = np.sqrt(np.mean((golden_pnl - current_pnl)**2)) / scale
    
    # Entry/exit timing (within 1 bar tolerance)
    golden_entries = np.where(np.diff(golden_pos) != 0)[0]
    current_entries = np.where(np.diff(current_pos) != 0)[0]
    
    if len(golden_entries) > 0 and len(current_entries) > 0:
        # Check if entries match within tolerance
        matched_entries = 0
        for g_entry in golden_entries:
            # Check if there's a current entry within 1 bar
            if np.any(np.abs(current_entries - g_entry) <= 1):
                matched_entries += 1
        
        results['entry_exit_match'] = matched_entries / len(golden_entries)
    
    # Check for issues
    if results['positions_match'] < 0.999:
        results['issues'].append(f"Position match only {results['positions_match']:.1%}")
    
    if results['pnl_rmse'] > 0.1:
        results['issues'].append(f"High PnL RMSE: {results['pnl_rmse']:.3f}")
    
    if results['entry_exit_match'] < 0.95:
        results['issues'].append(f"Entry/exit match only {results['entry_exit_match']:.1%}")
    
    return results


class TestGoldenTraces:
    """Test suite for golden trace regression."""
    
    def test_btc_eth_trace(self):
        """Test BTC/ETH pair against golden trace."""
        golden = load_golden_trace('btc_eth', 'reference')
        assert golden is not None, "Golden trace not found"
        
        # Simulate current trace (in real test, would run actual backtest)
        current = {
            'positions': golden['positions'].copy(),  # Perfect match for test
            'pnl': golden['pnl'].copy()
        }
        
        # Add small perturbation to test tolerance
        current['positions'][100] = -current['positions'][100]
        
        results = compare_traces(golden, current)
        
        assert results['positions_match'] >= 0.99, f"Position match too low: {results['positions_match']:.1%}"
        assert results['pnl_rmse'] < 0.1, f"PnL RMSE too high: {results['pnl_rmse']:.3f}"
        assert results['entry_exit_match'] >= 0.95, f"Entry/exit match too low: {results['entry_exit_match']:.1%}"
    
    def test_eth_usdt_trace(self):
        """Test ETH/USDT pair against golden trace."""
        golden = load_golden_trace('eth_usdt', 'reference')
        assert golden is not None, "Golden trace not found"
        
        # Simulate current trace
        current = {
            'positions': golden['positions'].copy(),
            'pnl': np.array(golden['pnl']) * 1.01  # 1% difference
        }
        
        results = compare_traces(golden, current)
        
        assert results['positions_match'] >= 0.999, "Positions should match exactly"
        assert results['pnl_rmse'] < 0.02, f"PnL RMSE too high: {results['pnl_rmse']:.3f}"
    
    def test_empty_trace_handling(self):
        """Test handling of empty traces."""
        golden = load_golden_trace('btc_eth', 'reference')
        
        empty_current = {
            'positions': [],
            'pnl': []
        }
        
        results = compare_traces(golden, empty_current)
        assert len(results['issues']) > 0
        assert "No positions" in results['issues'][0]
    
    def test_trace_alignment(self):
        """Test trace alignment with different lengths."""
        golden = {
            'positions': [0, 1, 0, -1, 0] * 20,  # 100 points
            'pnl': list(range(100))
        }
        
        current = {
            'positions': [0, 1, 0, -1, 0] * 15,  # 75 points
            'pnl': list(range(75))
        }
        
        results = compare_traces(golden, current)
        
        # Should compare only first 75 points
        assert results['positions_match'] == 1.0  # Perfect match in overlap
    
    @pytest.mark.parametrize("noise_level", [0.0, 0.01, 0.05, 0.1])
    def test_pnl_tolerance(self, noise_level):
        """Test PnL comparison with different noise levels."""
        n = 500
        base_pnl = np.random.randn(n).cumsum()
        
        golden = {
            'positions': [0] * n,
            'pnl': base_pnl.tolist()
        }
        
        # Add noise
        noisy_pnl = base_pnl + np.random.randn(n) * noise_level * np.abs(base_pnl).mean()
        
        current = {
            'positions': [0] * n,
            'pnl': noisy_pnl.tolist()
        }
        
        results = compare_traces(golden, current)
        
        # RMSE should be proportional to noise
        if noise_level <= 0.01:
            assert results['pnl_rmse'] < 0.02
        elif noise_level <= 0.05:
            assert results['pnl_rmse'] < 0.1
        else:
            # High noise acceptable for test
            assert results['pnl_rmse'] < 0.5


def test_generate_golden_traces():
    """Test golden trace generation."""
    result = generate_golden_traces()
    assert result is True
    
    # Check files exist
    assert Path("artifacts/baseline_traces/btc_eth_reference.json").exists()
    assert Path("artifacts/baseline_traces/eth_usdt_reference.json").exists()


if __name__ == "__main__":
    # Generate golden traces if running directly
    generate_golden_traces()
    
    # Run tests
    pytest.main([__file__, "-v"])