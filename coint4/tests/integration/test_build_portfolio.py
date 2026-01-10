#!/usr/bin/env python3
"""Integration tests for build_portfolio.py end-to-end workflow."""

import pytest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import tempfile
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from coint2.portfolio.optimizer import PortfolioConfig


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for integration tests."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create necessary subdirectories
    (temp_dir / 'bench').mkdir()
    (temp_dir / 'configs').mkdir()
    (temp_dir / 'artifacts' / 'portfolio').mkdir(parents=True)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_universe(temp_workspace):
    """Create sample universe file."""
    universe_data = {
        'pairs': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'DOT/USDT', 'LINK/USDT'],
        'metadata': {
            'last_updated': '2024-08-10',
            'total_pairs': 6
        }
    }
    
    universe_file = temp_workspace / 'bench' / 'pairs_universe.yaml'
    with open(universe_file, 'w') as f:
        yaml.dump(universe_data, f)
    
    return universe_file


@pytest.fixture
def sample_config(temp_workspace):
    """Create sample portfolio config."""
    config_data = {
        'selection': {
            'method': 'score_topN',
            'top_n': 4,
            'min_pairs': 3
        },
        'scoring': {
            'alpha_fee': 0.5,
            'beta_slip': 0.3
        },
        'optimizer': {
            'lambda_var': 1.0,
            'gamma_cost': 0.5,
            'max_gross': 1.0,
            'max_weight_per_pair': 0.25
        },
        'fallback': 'vol_target',
        'seed': 42
    }
    
    config_file = temp_workspace / 'configs' / 'portfolio_optimizer.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    return config_file


def test_build_portfolio_integration(temp_workspace, sample_universe, sample_config):
    """Test complete build_portfolio.py workflow."""
    
    # Import the build_portfolio module
    import importlib.util
    script_path = Path(__file__).parent.parent.parent / 'scripts' / 'build_portfolio.py'
    spec = importlib.util.spec_from_file_location("build_portfolio", script_path)
    build_module = importlib.util.module_from_spec(spec)
    
    # Change to temp workspace
    original_cwd = Path.cwd()
    try:
        import os
        os.chdir(temp_workspace)
        
        # Execute main components
        spec.loader.exec_module(build_module)
        
        # Load universe
        pairs = build_module.load_universe(sample_universe)
        assert len(pairs) == 6
        assert 'BTC/USDT' in pairs
        
        # Load metrics (will create synthetic data)
        metrics_df = build_module.load_metrics_from_artifacts(pairs, {})
        assert len(metrics_df) == 6
        assert 'psr' in metrics_df.columns
        assert 'vol' in metrics_df.columns
        
        # Load configuration
        from coint2.portfolio.optimizer import load_config
        config = load_config(sample_config)
        assert config.top_n == 4
        assert config.method == "score_topN"
        
        # Run optimization
        from coint2.portfolio.optimizer import PortfolioOptimizer
        optimizer = PortfolioOptimizer(config)
        result = optimizer.optimize_portfolio(metrics_df)
        
        assert result.success
        assert len(result.selected_pairs) <= config.top_n
        assert len(result.weights) > 0
        
        # Test save outputs
        output_dir = temp_workspace / 'artifacts' / 'portfolio'
        portfolio_path, weights_path = build_module.save_portfolio_outputs(
            {
                'weights': result.weights,
                'selected_pairs': result.selected_pairs,
                'diagnostics': result.diagnostics
            },
            pairs,
            metrics_df,
            output_dir
        )
        
        # Verify files were created
        assert portfolio_path.exists()
        assert weights_path.exists()
        assert (output_dir / 'PORTFOLIO_REPORT.md').exists()
        assert (output_dir / 'optimizer_diag.json').exists()
        
        # Verify content
        with open(portfolio_path, 'r') as f:
            portfolio_data = yaml.safe_load(f)
        
        assert 'pairs' in portfolio_data
        assert 'metadata' in portfolio_data
        assert len(portfolio_data['pairs']) == len(result.selected_pairs)
        
        weights_df = pd.read_csv(weights_path)
        assert 'pair' in weights_df.columns
        assert 'weight' in weights_df.columns
        assert len(weights_df) == len(result.selected_pairs)
        
    finally:
        os.chdir(original_cwd)


def test_wfa_integration_smoke():
    """Smoke test for WFA integration with portfolio parameters."""
    
    # Test that WFA script can handle portfolio parameters without crashing
    script_path = Path(__file__).parent.parent.parent / 'scripts' / 'run_walk_forward.py'
    
    # Import WFA module 
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_walk_forward", script_path)
    wfa_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wfa_module)
    
    # Check that _apply_portfolio_config method exists
    assert hasattr(wfa_module.WalkForwardAnalyzer, '_apply_portfolio_config')
    
    # Create mock analyzer
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'data_source': 'mock'}, f)
        config_path = f.name
    
    try:
        analyzer = wfa_module.WalkForwardAnalyzer(config_path)
        
        # Test that method can be called without error
        analyzer._apply_portfolio_config(None, None)
        
        # Should have portfolio_info attribute after call
        assert hasattr(analyzer, 'portfolio_info')
        
    finally:
        Path(config_path).unlink()


def test_paper_week_integration_smoke():
    """Smoke test for paper week integration."""
    
    script_path = Path(__file__).parent.parent.parent / 'scripts' / 'run_paper_week.py'
    
    # Import paper week module
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_paper_week", script_path)
    paper_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paper_module)
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / 'artifacts' / 'live' / 'metrics').mkdir(parents=True)
        
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_path)
            
            # Test function can run with portfolio parameters
            result = paper_module.run_paper_week(pairs_file=None, weights_file=None)
            
            # Should return DataFrame
            assert isinstance(result, pd.DataFrame)
            assert len(result) >= 0  # May be empty but should not fail
            
            # Check that artifacts were created
            assert (temp_path / 'artifacts' / 'live' / 'WEEKLY_SUMMARY.md').exists()
            
        finally:
            os.chdir(original_cwd)


def test_ci_gates_portfolio_validation_smoke():
    """Smoke test for CI gates portfolio validation."""
    
    script_path = Path(__file__).parent.parent.parent / 'scripts' / 'ci_gates.py'
    
    # Import CI gates module
    import importlib.util
    spec = importlib.util.spec_from_file_location("ci_gates", script_path)
    ci_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ci_module)
    
    # Test with mock config
    mock_config = {
        'portfolio_gates': {
            'enabled': False  # Disabled to avoid file dependencies
        }
    }
    
    checker = ci_module.CIGateChecker(mock_config, verbose=False)
    
    # Should handle disabled case gracefully
    passed, result = checker.check_portfolio_gates()
    
    assert passed  # Should pass when disabled
    assert 'message' in result


def test_full_pipeline_simulation():
    """Simulate full pipeline: universe -> portfolio -> WFA/weekly -> CI."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Setup directory structure
        (temp_path / 'bench').mkdir()
        (temp_path / 'configs').mkdir()
        (temp_path / 'artifacts' / 'portfolio').mkdir(parents=True)
        (temp_path / 'artifacts' / 'live' / 'metrics').mkdir(parents=True)
        
        # 1. Create universe
        universe_data = {
            'pairs': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
            'metadata': {'total_pairs': 3}
        }
        
        universe_file = temp_path / 'bench' / 'pairs_universe.yaml'
        with open(universe_file, 'w') as f:
            yaml.dump(universe_data, f)
        
        # 2. Create portfolio config
        config_data = {
            'selection': {'method': 'score_topN', 'top_n': 3, 'min_pairs': 2},
            'optimizer': {'max_gross': 1.0, 'max_weight_per_pair': 0.4},
            'fallback': 'vol_target'
        }
        
        config_file = temp_path / 'configs' / 'portfolio_optimizer.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # 3. Simulate portfolio construction
        from coint2.portfolio.optimizer import load_config, PortfolioOptimizer
        
        config = load_config(config_file)
        optimizer = PortfolioOptimizer(config)
        
        # Create mock metrics
        pairs = universe_data['pairs']
        metrics_df = pd.DataFrame({
            'exp_return': [0.08, 0.07, 0.06],
            'vol': [0.20, 0.18, 0.22],
            'psr': [0.7, 0.6, 0.5],
            'est_fee_per_turnover': [0.001, 0.001, 0.001],
            'est_slippage_per_turnover': [0.0005, 0.0005, 0.0005],
            'turnover_baseline': [0.1, 0.1, 0.1],
            'adv_proxy': [1000000, 800000, 600000],
            'cap_per_pair': [0.15, 0.15, 0.15]
        }, index=pairs)
        
        result = optimizer.optimize_portfolio(metrics_df)
        assert result.success
        
        # 4. Save portfolio outputs
        portfolio_data = {
            'pairs': result.selected_pairs,
            'metadata': {
                'total_selected': len(result.selected_pairs),
                'gross_exposure': result.diagnostics.get('gross_exposure', 1.0),
                'max_weight': result.diagnostics.get('max_weight', 0.33)
            }
        }
        
        pairs_portfolio = temp_path / 'bench' / 'pairs_portfolio.yaml'
        with open(pairs_portfolio, 'w') as f:
            yaml.dump(portfolio_data, f)
        
        weights_df = pd.DataFrame({
            'pair': result.weights.index,
            'weight': result.weights.values
        })
        weights_file = temp_path / 'artifacts' / 'portfolio' / 'weights.csv'
        weights_df.to_csv(weights_file, index=False)
        
        # Create portfolio report
        report_content = """# Portfolio Report
## Portfolio Metrics
- Selected Pairs: 3
- Gross Exposure: 1.000
- Max Weight: 0.333

## Capacity Check  
| Pair | Status |
|------|--------|
| BTC/USDT | ✅ OK |
| ETH/USDT | ✅ OK |
| ADA/USDT | ✅ OK |
"""
        report_file = temp_path / 'artifacts' / 'portfolio' / 'PORTFOLIO_REPORT.md'
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        # 5. Test CI gates can validate portfolio
        ci_config = {
            'portfolio_gates': {
                'enabled': True,
                'require_pairs_file': str(pairs_portfolio),
                'weights_file': str(weights_file),
                'portfolio_report': str(report_file),
                'min_pairs': 2,
                'max_weight_per_pair': 0.5,
                'max_gross': 1.1
            }
        }
        
        # Import CI module
        import importlib.util
        script_path = Path(__file__).parent.parent.parent / 'scripts' / 'ci_gates.py'
        spec = importlib.util.spec_from_file_location("ci_gates", script_path)
        ci_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ci_module)
        
        checker = ci_module.CIGateChecker(ci_config, verbose=False)
        passed, result = checker.check_portfolio_gates()
        
        # Should pass validation
        assert passed, f"Portfolio validation failed: {result.get('failures', [])}"
        assert 'metrics' in result
        assert result['metrics']['selected_pairs'] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])