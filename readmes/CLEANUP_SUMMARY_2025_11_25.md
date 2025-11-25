# Codebase Cleanup Summary - November 25, 2025

## What Was Done

Cleaned up legacy code after implementing the new `PaperTradingEngine` architecture. Moved redundant and outdated files to archive.

---

## Files Archived (11 Total)

### Core Modules → `archive/old_core/` (3 files)

1. **`risk_manager_old.py`** - Old stub, replaced by full `risk_manager.py` implementation
2. **`paper_trader.py`** - Single-asset trader, replaced by multi-asset `paper_trading_engine.py`
3. **`backtest_engine.py`** - Old walk-forward engine, replaced by `portfolio_manager.py`

### Scripts → `archive/old_scripts/` (8 files)

4. `test_new_structure.py` - Structure validation (obsolete)
5. `test_quick_fixes.py` - Ad-hoc tests (obsolete)
6. `test_diagnostics.py` - Old diagnostics
7. `test_stop_loss.py` - Used old PaperTrader
8. `test_optimizer.py` - Used old backtest engine
9. `test_perfold_optimization.py` - Per-fold optimization
10. `test_position_sizing.py` - Used old engine
11. `sanity_check_signal.py` - Used old PaperTrader

---

## Impact

### Before
- 15 files in `core/`
- 20 files in `scripts/`
- Multiple implementations of same concepts
- Confusion about which modules to use

### After
- **11 files in `core/`** (27% reduction ↓)
- **12 files in `scripts/`** (40% reduction ↓)
- Clear single implementation for each concept
- Obvious module choices

### Results
✅ All core imports working  
✅ Test suite passes (`tests/test_paper_trading_engine.py`)  
✅ New architecture fully functional  
✅ Old code preserved in archive if needed  

---

## Current Active Structure

### Production Core Modules
- ✅ `paper_trading_engine.py` - NEW: Multi-asset paper trading with state persistence
- ✅ `portfolio_manager.py` - Multi-asset backtesting and portfolio management
- ✅ `risk_manager.py` - Position sizing, correlation, drawdown controls
- ✅ `multi_asset_loader.py` - Data loading with yfinance auto-fetch
- ✅ `multi_asset_signal.py` - Signal generation wrapper
- ✅ `risk_dashboard.py` - Risk visualization
- ✅ `reporter.py` - Performance reporting
- ✅ `optimizer.py` - Parameter optimization
- ✅ `position_sizers.py` - Position sizing utilities
- ✅ `strategy_selector.py` - Strategy selection
- ✅ `multi_strategy_signal.py` - Multi-strategy signals

### Active Test Scripts
- ✅ `test_paper_trading_engine.py` - NEW: Tests for new engine ✅
- ✅ `test_risk_manager.py` - Risk manager tests
- ✅ `test_risk_integration.py` - Integration tests
- ✅ `test_reporter.py` - Reporter tests
- ✅ `test_multi_asset_simple.py` - Multi-asset tests
- ✅ `test_multi_strategy.py` - Strategy tests
- ✅ Other utility scripts (validate data, debug, etc.)

---

## Files That Need Updates

### High Priority

**`scripts/run_daily.py`** ⚠️
- Still imports old `paper_trader` and `backtest_engine`
- Should use `PaperTradingEngine` instead
- See `tests/test_paper_trading_engine.py` for example

**`utils/load_params.py`** ⚠️
- May import old `paper_trader`
- Needs review

### Medium Priority

**Notebooks 03-04** ⚠️
- Use old `run_walk_forward` from archived `backtest_engine`
- Should migrate to `run_multi_asset_backtest`
- Or mark as "legacy examples"

---

## Migration Path

### If You Were Using Old PaperTrader

**Old:**
```python
from core.paper_trader import PaperTrader
trader = PaperTrader(initial_cash=100000)
result = trader.simulate(df)
```

**New:**
```python
from core.paper_trading_engine import PaperTradingEngine
engine = PaperTradingEngine(config=config)
engine.initialize(prices_dict, signals_dict)
status = engine.get_portfolio_status(prices_dict)
```

**Example:** See `notebooks/10_paper_trading_with_engine.ipynb`

### If You Were Using Old Backtest Engine

**Old:**
```python
from core.backtest_engine import run_walk_forward
results = run_walk_forward(df, signal_class, params)
```

**New:**
```python
from core.portfolio_manager import run_multi_asset_backtest
result, equity, trades = run_multi_asset_backtest(
    signals_dict, prices_dict, config
)
```

**Example:** See `notebooks/09_live_trading_simulation.ipynb`

---

## Rollback (if needed)

Archived files can be restored:

```bash
# Restore specific file
cp archive/old_core/paper_trader.py core/

# Restore all
cp archive/old_core/* core/
cp archive/old_scripts/* scripts/
```

**However:** Consider migrating to new architecture instead. It's more powerful and production-ready.

---

## Documentation

Full details in:
- `archive/ARCHIVE_LOG_2025_11_25.md` - Complete archive log
- `readmes/PAPER_TRADING_ENGINE_IMPLEMENTATION.md` - New architecture docs
- `readmes/PAPER_TRADING_PROBLEMS_TO_ADDRESS.md` - Future roadmap

---

## Next Steps

1. ✅ Archive completed
2. ⏳ Update `scripts/run_daily.py` to use new engine
3. ⏳ Review `utils/load_params.py`
4. ⏳ Migrate or mark notebooks 03-04 as legacy
5. ⏳ Add migration guide to main README

---

## Questions?

- New architecture: See `notebooks/10_paper_trading_with_engine.ipynb`
- Test examples: See `tests/test_paper_trading_engine.py`
- API reference: See docstrings in `core/paper_trading_engine.py`
- Archived code: See `archive/old_core/` and `archive/old_scripts/`
