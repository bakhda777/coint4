"""Tests for portfolio allocator."""

import pytest
import sys
sys.path.insert(0, 'src')

from coint2.portfolio import (
    Signal, 
    EqualWeightAllocator,
    VolTargetAllocator,
    RiskParityAllocator,
    CapPerPairAllocator,
    create_allocator
)


def test_equal_weight_allocator():
    """Test equal weight allocation."""
    config = {
        'max_weight_per_pair': 0.20,
        'max_gross': 1.0,
        'max_net': 0.5
    }
    
    allocator = EqualWeightAllocator(config)
    
    signals = [
        Signal('BTC/ETH', 1.0, 0.8, 0.3, 1.5),
        Signal('ETH/USDT', -1.0, 0.7, 0.4, 1.2),
        Signal('BNB/USDT', 1.0, 0.6, 0.35, 1.0),
    ]
    
    result = allocator.allocate(signals)
    
    # Check weights sum
    assert len(result.weights) == 3
    assert abs(result.gross_exposure - sum(abs(w) for w in result.weights.values())) < 1e-6
    
    # Check constraints
    for weight in result.weights.values():
        assert abs(weight) <= config['max_weight_per_pair'] + 1e-6
    
    assert result.gross_exposure <= config['max_gross'] + 1e-6
    assert abs(result.net_exposure) <= config['max_net'] + 1e-6


def test_vol_target_allocator():
    """Test volatility targeting allocation."""
    config = {
        'target_vol': 0.10,
        'max_weight_per_pair': 0.30,
        'max_gross': 1.0,
        'max_net': 0.5
    }
    
    allocator = VolTargetAllocator(config)
    
    signals = [
        Signal('BTC/ETH', 1.0, 0.8, 0.2, 1.5),  # Low vol
        Signal('ETH/USDT', -1.0, 0.7, 0.5, 1.2),  # High vol
        Signal('BNB/USDT', 1.0, 0.6, 0.3, 1.0),  # Med vol
    ]
    
    result = allocator.allocate(signals)
    
    # Lower vol should get higher weight
    assert abs(result.weights['BTC/ETH']) > abs(result.weights['ETH/USDT'])
    
    # Check constraints
    assert result.gross_exposure <= config['max_gross'] + 1e-6
    assert abs(result.net_exposure) <= config['max_net'] + 1e-6


def test_risk_parity_allocator():
    """Test risk parity allocation."""
    config = {
        'max_weight_per_pair': 0.25,
        'max_gross': 1.0,
        'max_net': 0.4
    }
    
    allocator = RiskParityAllocator(config)
    
    signals = [
        Signal('BTC/ETH', 1.0, 0.9, 0.2, 2.0),  # High Sharpe, low vol
        Signal('ETH/USDT', 1.0, 0.5, 0.4, 0.5),  # Low Sharpe, high vol
        Signal('BNB/USDT', -1.0, 0.7, 0.3, 1.5),  # Good Sharpe, med vol
    ]
    
    result = allocator.allocate(signals)
    
    # Should allocate based on risk-adjusted score
    assert len(result.weights) == 3
    assert result.n_positions > 0
    
    # Check constraints
    for weight in result.weights.values():
        assert abs(weight) <= config['max_weight_per_pair'] + 1e-6


def test_cap_per_pair_allocator():
    """Test capped allocation."""
    config = {
        'max_weight_per_pair': 0.10,
        'max_gross': 0.50,
        'max_net': 0.30
    }
    
    allocator = CapPerPairAllocator(config)
    
    signals = [
        Signal('BTC/ETH', 1.0, 0.9, 0.3, 2.0),
        Signal('ETH/USDT', 1.0, 0.8, 0.3, 1.8),
        Signal('BNB/USDT', 1.0, 0.7, 0.3, 1.6),
        Signal('XRP/USDT', 1.0, 0.6, 0.3, 1.4),
        Signal('ADA/USDT', 1.0, 0.5, 0.3, 1.2),
    ]
    
    result = allocator.allocate(signals)
    
    # Should respect cap
    for weight in result.weights.values():
        assert abs(weight) <= config['max_weight_per_pair'] + 1e-6
    
    # Should fill up to gross limit
    assert result.gross_exposure <= config['max_gross'] + 1e-6
    
    # Net should be within limits
    assert abs(result.net_exposure) <= config['max_net'] + 1e-6


def test_empty_signals():
    """Test allocators with empty signals."""
    config = {'max_weight_per_pair': 0.10, 'max_gross': 1.0, 'max_net': 0.5}
    
    allocators = [
        EqualWeightAllocator(config),
        VolTargetAllocator(config),
        RiskParityAllocator(config),
        CapPerPairAllocator(config)
    ]
    
    for allocator in allocators:
        result = allocator.allocate([])
        assert len(result.weights) == 0
        assert result.gross_exposure == 0
        assert result.net_exposure == 0
        assert result.n_positions == 0


def test_net_exposure_constraint():
    """Test net exposure constraint enforcement."""
    config = {
        'max_weight_per_pair': 0.50,
        'max_gross': 2.0,
        'max_net': 0.20  # Tight net constraint
    }
    
    allocator = EqualWeightAllocator(config)
    
    # All long signals
    signals = [
        Signal('BTC/ETH', 1.0, 0.9, 0.3, 1.5),
        Signal('ETH/USDT', 1.0, 0.8, 0.3, 1.2),
        Signal('BNB/USDT', 1.0, 0.7, 0.3, 1.0),
    ]
    
    result = allocator.allocate(signals)
    
    # Net exposure should be capped
    assert result.net_exposure <= config['max_net'] + 1e-6


def test_create_allocator_factory():
    """Test allocator factory function."""
    methods = ['equal_weight', 'vol_target', 'risk_parity', 'cap_per_pair']
    
    for method in methods:
        config = {'method': method}
        allocator = create_allocator(config)
        assert allocator is not None
        
    # Test unknown method
    with pytest.raises(ValueError):
        create_allocator({'method': 'unknown'})


def test_mixed_long_short_signals():
    """Test allocation with mixed long/short signals."""
    config = {
        'max_weight_per_pair': 0.20,
        'max_gross': 1.0,
        'max_net': 0.10  # Tight net constraint for market neutral
    }
    
    allocator = EqualWeightAllocator(config)
    
    signals = [
        Signal('BTC/ETH', 1.0, 0.8, 0.3, 1.5),   # Long
        Signal('ETH/USDT', -1.0, 0.7, 0.3, 1.2),  # Short
        Signal('BNB/USDT', 1.0, 0.6, 0.3, 1.0),   # Long
        Signal('XRP/USDT', -1.0, 0.5, 0.3, 0.8),  # Short
    ]
    
    result = allocator.allocate(signals)
    
    # Should have balanced long/short
    longs = sum(w for w in result.weights.values() if w > 0)
    shorts = abs(sum(w for w in result.weights.values() if w < 0))
    
    # Net should be small due to constraint
    assert abs(result.net_exposure) <= config['max_net'] + 1e-6
    
    # Gross should be sum of abs weights
    assert abs(result.gross_exposure - (longs + shorts)) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])