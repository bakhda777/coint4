#!/usr/bin/env python3
"""Test CLI module imports and structure."""

import pytest
import importlib
import sys
from pathlib import Path


def test_cli_package_imports():
    """Test that CLI package imports correctly."""
    # Test main package import
    import coint2.cli
    assert hasattr(coint2.cli, '__all__')
    assert 'check_coint_health' in coint2.cli.__all__
    assert 'build_universe' in coint2.cli.__all__


def test_check_coint_health_imports():
    """Test check_coint_health module imports."""
    from coint2.cli import check_coint_health
    
    # Test main function exists
    assert hasattr(check_coint_health, 'main')
    assert callable(check_coint_health.main)
    
    # Test key functions
    assert hasattr(check_coint_health, 'check_coint_health')
    assert hasattr(check_coint_health, 'check_pair_health')
    assert hasattr(check_coint_health, 'generate_health_report')


def test_build_universe_imports():
    """Test build_universe module imports."""
    from coint2.cli import build_universe
    
    # Test main function exists
    assert hasattr(build_universe, 'main')
    assert callable(build_universe.main)
    
    # Test key functions
    assert hasattr(build_universe, 'build_universe')
    assert hasattr(build_universe, 'apply_selection_rules')
    assert hasattr(build_universe, 'generate_report')
    assert hasattr(build_universe, 'save_outputs')


def test_no_sys_path_hacks():
    """Ensure CLI modules don't use sys.path hacks."""
    cli_dir = Path(__file__).parent.parent.parent / 'src/coint2/cli'
    
    for py_file in cli_dir.glob('*.py'):
        if py_file.name == '__init__.py':
            continue
            
        with open(py_file) as f:
            content = f.read()
            
        # Check for sys.path manipulations
        assert 'sys.path.insert' not in content, f"Found sys.path.insert in {py_file.name}"
        assert 'sys.path.append' not in content, f"Found sys.path.append in {py_file.name}"


def test_cli_modules_have_proper_imports():
    """Test that CLI modules use proper package imports."""
    from coint2.cli import check_coint_health, build_universe
    
    # These should work without ImportError
    assert check_coint_health.DataHandler
    assert check_coint_health.load_config
    assert check_coint_health.test_cointegration
    assert check_coint_health.estimate_half_life
    
    assert build_universe.DataHandler
    assert build_universe.load_config
    assert build_universe.scan_universe
    assert build_universe.calculate_pair_score


def test_entry_points_callable():
    """Test that entry points are callable without errors."""
    from coint2.cli.check_coint_health import main as check_health_main
    from coint2.cli.build_universe import main as build_universe_main
    
    # Verify they're callable
    assert callable(check_health_main)
    assert callable(build_universe_main)
    
    # Check they have argparse setup (won't actually run them)
    import inspect
    
    # Should accept no arguments (argparse handles sys.argv)
    check_sig = inspect.signature(check_health_main)
    assert len(check_sig.parameters) == 0
    
    build_sig = inspect.signature(build_universe_main)
    assert len(build_sig.parameters) == 0


def test_module_execution_mode():
    """Test modules can be executed with python -m."""
    import subprocess
    import sys
    import os
    from pathlib import Path
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    
    # Test check_coint_health help
    result = subprocess.run(
        [sys.executable, '-m', 'coint2.cli.check_coint_health', '--help'],
        capture_output=True,
        text=True,
        env=env
    )
    assert result.returncode == 0
    assert 'Check cointegration health' in result.stdout
    
    # Test build_universe help
    result = subprocess.run(
        [sys.executable, '-m', 'coint2.cli.build_universe', '--help'],
        capture_output=True,
        text=True,
        env=env
    )
    assert result.returncode == 0
    assert 'Build universe of cointegrated pairs' in result.stdout
