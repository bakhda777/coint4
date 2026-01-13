> NOTE: Archived/legacy document. It may describe historical behavior and can be out of date. See `docs/` for current usage.

# Release Notes - v0.1.8.1

## Import Issues Fixed & CLI Unified

### What Was Fixed

#### 1. Import Errors ✅
- **Fixed**: `ImportError: cannot import name 'BaseEngine'` in engine module
- **Fixed**: `ImportError: cannot import name 'PairScanner'` in pipeline module  
- **Fixed**: `ImportError: cannot import name 'PairFilter'` in pipeline module

These were caused by `__init__.py` files trying to import non-existent classes. The modules contained functions, not classes.

#### 2. CLI Package Structure ✅
- **Created**: Proper CLI package at `src/coint2/cli/`
- **Migrated**: Scripts from `scripts/` to CLI package
- **Removed**: All `sys.path.insert(0, ...)` hacks
- **Added**: Proper ImportError handling with user guidance

#### 3. Entry Points ✅
Now you can use these commands after `poetry install`:
```bash
coint2-check-health --pairs-file bench/pairs_universe.yaml
coint2-build-universe --config configs/universe.yaml
```

Or use module execution:
```bash
python -m coint2.cli.check_coint_health --help
python -m coint2.cli.build_universe --help
```

#### 4. Scheduler Hardening ✅
The scheduler now uses proper module calls:
```bash
python -m coint2.cli.check_coint_health
python -m coint2.cli.build_universe
```

### Testing

All import tests pass:
```bash
pytest tests/unit/test_cli_imports.py -v
# ============================== 7 passed in 11.95s ==============================
```

### Files Changed

#### Created
- `src/coint2/cli/__init__.py`
- `src/coint2/cli/check_coint_health.py`
- `src/coint2/cli/build_universe.py`
- `tests/unit/test_cli_imports.py`

#### Modified
- `src/coint2/engine/__init__.py` - Fixed class imports
- `src/coint2/pipeline/__init__.py` - Fixed class imports
- `pyproject.toml` - Added CLI entry points
- `scripts/scheduler_local.py` - Use module calls
- `CHANGELOG.md` - Added v0.1.8.1 section

### How to Use

1. **Install the package**:
   ```bash
   poetry install
   ```

2. **Use CLI commands**:
   ```bash
   # Check cointegration health
   coint2-check-health --pairs-file bench/pairs_universe.yaml
   
   # Build universe
   coint2-build-universe --config configs/universe.yaml
   ```

3. **Or use module execution**:
   ```bash
   python -m coint2.cli.check_coint_health
   python -m coint2.cli.build_universe
   ```

4. **Run scheduler**:
   ```bash
   python scripts/scheduler_local.py --daily
   python scripts/scheduler_local.py --weekly --rebuild-universe
   ```

### Benefits

✅ **No more import errors** - Proper package structure  
✅ **Clean code** - No sys.path hacks  
✅ **Professional CLI** - Entry points for easy commands  
✅ **Better testing** - Can properly test imports  
✅ **Maintainable** - Clear package organization  

### Version

Updated from v0.1.8 to **v0.1.8.1** - Patch release for import fixes and CLI unification.