"""Smoke tests for package imports."""

import pytest


@pytest.mark.smoke
def test_coint2_package_import():
    """Test that coint2 package can be imported."""
    import coint2
    assert coint2.__version__ == "0.1.0"


@pytest.mark.smoke  
def test_core_modules_import():
    """Test that core modules can be imported."""
    from coint2.utils.config import load_config, AppConfig
    from coint2.core.data_loader import DataHandler
    from coint2.pipeline.walk_forward_orchestrator import WalkForwardOrchestrator
    
    # Check exports
    assert load_config is not None
    assert AppConfig is not None
    assert DataHandler is not None
    assert WalkForwardOrchestrator is not None


@pytest.mark.smoke
def test_cli_entry_points():
    """Test that CLI entry points are importable."""
    from coint2.cli import main as cli_main
    from coint2.cli_live import main as live_main
    
    assert cli_main is not None
    assert live_main is not None


@pytest.mark.smoke
def test_engine_imports():
    """Test that backtesting engines can be imported."""
    from coint2.engine.reference_engine import ReferenceEngine
    from coint2.engine.base_engine import BaseEngine
    
    assert ReferenceEngine is not None
    assert BaseEngine is not None