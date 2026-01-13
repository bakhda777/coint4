> NOTE: Archived/legacy document. It may describe historical behavior and can be out of date. See `docs/` for current usage.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
- **Smoke tests (< 10 seconds)**: `./scripts/test_smoke.sh` - Critical fast tests only
- **Fast tests (< 1 minute)**: `./scripts/test_fast.sh` - CI/CD suitable tests, runs parallel then serial
- **Unit tests**: `./scripts/test_unit_fast.sh` - Isolated unit tests with mocks
- **Development tests**: `./scripts/test_dev.sh` - Broader test suite for development
- **Full test suite**: `./scripts/test_full.sh` - Complete test suite including slow tests
- **Single test**: `pytest tests/path/to/test_file.py::test_function -v`

### Code Quality
- **Linting**: `ruff check src/ tests/` - Code style and quality checks
- **Type checking**: `mypy src/` - Static type analysis
- **Auto-format**: `ruff format src/ tests/` - Automatic code formatting

### Optimization and Analysis
- **Quick optimization**: `python scripts/bp_optimize.py --n-trials 50 --fast`
- **Validation**: `python scripts/bp_validate.py`
- **Web analysis**: `python scripts/web_server.py` (for viewing optimization results)
- **Optuna dashboard**: `optuna-dashboard sqlite:///outputs/studies/[study_name].db`

### Main CLI Commands
- **Backtest pair**: `poetry run coint2 backtest --pair BTCUSDT,ETHUSDT`
- **Scan pairs**: `poetry run coint2 scan`
- **Full pipeline**: `poetry run coint2 run-pipeline`
- **Module execution**: `python -m coint2 run --config configs/main_2024.yaml`
- **Debug single pair**: Use `configs/debug_single_pair.yaml` for isolated pair analysis
- **Trade generation test**: Use `configs/test_trade_generation.yaml` for trade logic validation

### Environment Setup
- **Dependencies**: `poetry install` - Install all project dependencies
- **Development mode**: All scripts expect to be run from project root directory
- **Python version**: Requires Python 3.10+ (specified in pyproject.toml)

## Architecture Overview

This is a **cointegration pairs trading framework** for cryptocurrency markets with three main components:

### 1. Core Engine (`src/coint2/`)
- **Data Layer**: `core/data_loader.py` handles partitioned Parquet files (symbol/year/month structure)
- **Math Core**: `core/fast_coint.py` and `core/math_utils.py` provide cointegration testing and mathematical utilities
- **Backtest Engine**: `engine/numba_backtest_engine_full.py` - High-performance Numba-compiled backtesting
- **Portfolio Management**: `core/portfolio.py` handles position sizing and risk management

### 2. Optimization System (`src/optimiser/`)
- **Objective Function**: `fast_objective.py` - Optuna-optimized scoring (Sharpe ratio with penalties)
- **Parameter Search**: Optimizes 18+ parameters across pair selection, signals, risk management, and portfolio allocation
- **Best Practice Optimization**: `bp_optimize.py` uses Z-score hysteresis and advanced parameter validation

### 3. Analysis Pipeline (`src/coint2/pipeline/`)
- **Walk-Forward Orchestrator**: `walk_forward_orchestrator.py` manages time-series cross-validation
- **Pair Scanner**: `pair_scanner.py` finds cointegrated pairs using statistical tests
- **Filters**: `filters.py` applies various screening criteria (stationarity, mean reversion, etc.)

### Data Flow
1. **Scanning**: Load symbol universe → Create pair combinations → Apply cointegration tests → Filter by statistical criteria
2. **Optimization**: Walk-forward validation → Optuna hyperparameter search → Generate best configuration
3. **Backtesting**: Load historical data → Apply strategy rules → Calculate performance metrics
4. **Analysis**: Web interface for results visualization and performance degradation analysis

## Key Configuration Files

- **`configs/main_2024_trae.yaml`**: Current primary strategy configuration
- **`configs/search_space_fast.yaml`**: Optuna parameter search space (fast validation)
- **`configs/search_space_ultra_fast.yaml`**: Ultra-fast optimization for development (< 30 seconds)
- **`configs/debug_single_pair.yaml`**: Isolated pair analysis configuration
- **`configs/test_trade_generation.yaml`**: Trade logic validation configuration
- **`configs/archive/`**: Historical best configurations from optimization runs
- **`pyproject.toml`**: Poetry dependencies and tool configuration
- **`pytest.ini`**: Test markers and execution settings with Russian comments
- **`mypy.ini`**: Type checking configuration

## Testing Philosophy

The project uses a sophisticated test categorization system:
- **smoke**: Critical tests (< 5 seconds total) - must always pass
- **fast**: CI/CD tests (< 30 seconds total) - for continuous integration
- **slow**: Integration tests (> 30 seconds) - for thorough validation
- **serial**: Tests that cannot be parallelized (SQLite, global state)
- **unit**: Isolated tests with mocks - fastest feedback loop
- **integration**: Tests with real data/systems
- **performance**: Benchmarks and speed tests
- **memory**: Memory usage validation tests
- **cache**: Caching system tests
- **concurrent**: Concurrency and thread safety tests
- **critical_fixes**: Tests for critical bug fixes

Always use appropriate test markers. Tests without proper markers will be excluded from CI runs.

### Test Execution Patterns
- **Parallel execution**: Fast tests run with `pytest -n auto`, serial tests run separately
- **Default behavior**: Tests automatically exclude `slow` and `serial` markers unless explicitly specified
- **Test timeout**: Global 120-second timeout configured in pytest.ini
- **Russian comments**: Test markers and some configuration use Russian language (legacy from original development)

### Test Organization Structure
```
tests/
├── unit/           # Fast isolated tests (<1 sec each)
├── integration/    # Real data integration tests  
├── performance/    # Benchmarks and speed validation
├── fixtures/       # Shared test fixtures
└── conftest.py     # Global test configuration
```

## Performance Considerations

- **Numba Compilation**: Core mathematical functions are JIT-compiled for speed
- **Polars DataFrames**: Used for high-performance data manipulation
- **Dask Integration**: Handles large datasets through lazy evaluation
- **Memory Optimization**: Global rolling cache system prevents redundant calculations
- **Concurrent Optimization**: Optuna supports parallel trial execution

## Important Patterns

### Lookahead Bias Prevention
All backtesting logic must use data only up to time `t-1` for decisions at time `t`. This is enforced through code reviews and specific test cases.

### Parameter Validation
Use `src/optimiser/metric_utils.py` for parameter validation and normalization. Never modify parameters without going through the validation layer.

### Configuration Management
Always use the config system (`src/coint2/utils/config.py`) rather than hardcoded values. Configurations are validated against Pydantic schemas.

### Error Handling
Prefer explicit error handling with informative messages. Use the logging system (`src/coint2/utils/logger.py`) for debugging information.

## Data Structure

Market data is stored in partitioned Parquet format:
```
data_downloaded/
  year=2024/
    month=01/
      data_part_01.parquet  # Must contain: timestamp, close, symbol
```

Results are written to timestamped directories in `results/` and optimization studies are stored in SQLite databases under `outputs/studies/`.

### Dependency Management
- **Package manager**: Poetry (configured in pyproject.toml)
- **Core dependencies**: NumPy, Pandas, Polars, Numba for performance; Optuna for optimization
- **Development tools**: Ruff (linting/formatting), MyPy (type checking), pytest (testing)
- **Data processing**: Dask for large datasets, PyArrow for Parquet files
- **Statistical libraries**: SciPy, Statsmodels for econometric tests

## Code Quality and Architecture Principles

### Development Rules (from `rules/`)
1. **Code Changes Require Tests**: All code modifications in `src/` must include corresponding test updates
2. **Single Responsibility**: Functions should have one clear purpose - avoid "god functions"  
3. **No Magic Numbers**: Constants must be in `src/optimiser/constants.py` or config files
4. **Performance First**: Use vectorization (`numpy`, `pandas`) or JIT compilation (`numba`) for data operations
5. **Strict Typing**: All new code requires complete type hints and must pass `mypy` validation

### Test Quality Standards
- **Deterministic Tests**: Use global seed fixation in `conftest.py` (never duplicate in test code)
- **Lookahead Bias Prevention**: Logic at time `t` can only use data up to `t-1`
- **Test Naming**: Follow `test_feature_when_condition_then_result` format
- **Minimal Fixtures**: Use `tiny_prices_df` for unit tests, `small_prices_df` for integration
- **No Flaky Tests**: Red tests indicate code issues, not test issues - never weaken tests to make them pass

### Optimization Test Guidelines
- **Optuna Tests**: Use minimal trials (3-5), `RandomSampler`, and `tmp_path` storage
- **Numba/BLAS**: Limit thread counts with `os.environ` to prevent CPU overload during parallel testing
- **Serial Marking**: Mark SQLite and global cache tests as `@pytest.mark.serial`