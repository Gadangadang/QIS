# Archive Log - November 25, 2025

## Purpose
Cleaned up legacy code after implementing PaperTradingEngine architecture. Moved redundant and outdated files to archive to improve codebase maintainability.

---

## Files Archived

### Core Module Cleanup

#### 1. `core/risk_manager_old.py` → `archive/old_core/`
**Status:** ✅ REDUNDANT - Replaced by new implementation

**Reason:** 
- Old skeleton/stub implementation with TODO comments
- Completely superseded by `core/risk_manager.py` (implemented in Phase 1)
- No active imports found in codebase
- New risk_manager.py has full implementation with:
  - Position sizing methods (equal_weight, vol_adjusted, risk_parity)
  - Correlation monitoring
  - Drawdown controls
  - Violation tracking
  - Dashboard generation

**Impact:** None - no files import this

---

#### 2. `core/paper_trader.py` → `archive/old_core/`
**Status:** ⚠️ DEPRECATED - Replaced by PaperTradingEngine

**Reason:**
- Simple single-asset paper trading simulator
- Replaced by `core/paper_trading_engine.py` which provides:
  - Multi-asset support
  - State persistence
  - Performance comparison
  - Daily report generation
  - Production-ready workflow
- Old implementation lacks:
  - Multi-asset portfolio management
  - Risk management integration
  - State persistence
  - Clean API for automation

**Files that imported it:**
- `scripts/run_daily.py` (old script, should use new engine)
- `scripts/test_stop_loss.py` (archived)
- `utils/load_params.py` (legacy utility)
- `scripts/sanity_check_signal.py` (archived)
- `scripts/test_optimizer.py` (archived)
- Several archived notebooks (03, 04)

**Migration Path:**
- Any code using `PaperTrader` should migrate to `PaperTradingEngine`
- See `notebooks/10_paper_trading_with_engine.ipynb` for examples
- See `tests/test_paper_trading_engine.py` for API usage

---

#### 3. `core/backtest_engine.py` → `archive/old_core/`
**Status:** ⚠️ DEPRECATED - Replaced by portfolio_manager.py

**Reason:**
- Old walk-forward backtesting implementation
- Tightly coupled to `PaperTrader` (which is also archived)
- Replaced by `core/portfolio_manager.py` with `run_multi_asset_backtest()`
- New implementation provides:
  - Multi-asset portfolio management
  - Integrated risk management
  - Better separation of concerns
  - More flexible configuration

**Files that imported it:**
- `notebooks/03_backtest_momentum.ipynb` (uses `run_walk_forward`)
- `notebooks/04_position_sizing_optimization.ipynb` (uses `run_walk_forward`)
- Multiple test scripts (now archived)

**Migration Path:**
- Replace `run_walk_forward()` with `run_multi_asset_backtest()`
- See `notebooks/09_live_trading_simulation.ipynb` for examples
- Use `PaperTradingEngine` for live trading workflows

---

### Scripts Cleanup

All these scripts use the old `backtest_engine.py` and `paper_trader.py` modules that have been archived:

#### 4. `scripts/test_new_structure.py` → `archive/old_scripts/`
**Status:** ✅ OBSOLETE
**Reason:** Test script for old code structure, no longer relevant after restructure

#### 5. `scripts/test_quick_fixes.py` → `archive/old_scripts/`
**Status:** ✅ OBSOLETE
**Reason:** Ad-hoc test for quick fixes, functionality now covered by unit tests

#### 6. `scripts/test_diagnostics.py` → `archive/old_scripts/`
**Status:** ✅ OBSOLETE
**Reason:** Diagnostic script for old walk-forward implementation

#### 7. `scripts/test_stop_loss.py` → `archive/old_scripts/`
**Status:** ✅ OBSOLETE
**Reason:** Tests stop-loss with old PaperTrader, functionality now in portfolio_manager

#### 8. `scripts/test_optimizer.py` → `archive/old_scripts/`
**Status:** ✅ OBSOLETE
**Reason:** Tests old parameter optimizer, needs update for new architecture

#### 9. `scripts/test_perfold_optimization.py` → `archive/old_scripts/`
**Status:** ✅ OBSOLETE
**Reason:** Per-fold optimization using old walk-forward engine

#### 10. `scripts/test_position_sizing.py` → `archive/old_scripts/`
**Status:** ✅ OBSOLETE
**Reason:** Position sizing tests using old engine, now handled by risk_manager

#### 11. `scripts/sanity_check_signal.py` → `archive/old_scripts/`
**Status:** ✅ OBSOLETE
**Reason:** Signal sanity checks using old PaperTrader

---

## Files Kept (Active Use)

### Core Modules (Production)
- ✅ `core/paper_trading_engine.py` - NEW: Production paper trading engine
- ✅ `core/portfolio_manager.py` - Production multi-asset backtesting
- ✅ `core/risk_manager.py` - Production risk management
- ✅ `core/multi_asset_loader.py` - Data loading with yfinance integration
- ✅ `core/multi_asset_signal.py` - Signal generation
- ✅ `core/risk_dashboard.py` - Risk visualization
- ✅ `core/reporter.py` - Still used in notebook 06 and test_reporter.py
- ✅ `core/optimizer.py` - Parameter optimization (may need update)
- ✅ `core/position_sizers.py` - Position sizing utilities
- ✅ `core/strategy_selector.py` - Strategy selection logic
- ✅ `core/multi_strategy_signal.py` - Multi-strategy signal generation

### Scripts (Active)
- ✅ `scripts/test_paper_trading_engine.py` - NEW: Tests for new engine
- ✅ `scripts/test_reporter.py` - Reporter functionality tests
- ✅ `scripts/test_risk_manager.py` - Risk manager tests
- ✅ `scripts/test_risk_integration.py` - Risk integration tests
- ✅ `scripts/test_multi_asset_simple.py` - Multi-asset tests
- ✅ `scripts/test_multi_strategy.py` - Multi-strategy tests
- ✅ `scripts/test_portfolio_allocation.py` - Portfolio allocation tests
- ✅ `scripts/validate_futures_data.py` - Data validation
- ✅ `scripts/compare_backtest_performance.py` - Performance comparison
- ✅ `scripts/debug_portfolio.py` - Portfolio debugging
- ✅ `scripts/find_best_trial.py` - Find best optimization trial
- ✅ `scripts/test_gc_load.py` - Gold futures loading test
- ⚠️ `scripts/run_daily.py` - NEEDS UPDATE (still imports old paper_trader)

### Notebooks (Active)
- ✅ `notebooks/10_paper_trading_with_engine.ipynb` - NEW: Production workflow
- ✅ `notebooks/09_live_trading_simulation.ipynb` - Live trading demo
- ✅ `notebooks/08_risk_integration_demo.ipynb` - Risk integration
- ✅ `notebooks/07_risk_management_demo.ipynb` - Risk management
- ✅ `notebooks/06_multi_asset_portfolio_demo.ipynb` - Multi-asset demo
- ⚠️ `notebooks/05_multi_asset_demo.ipynb` - May need update
- ⚠️ `notebooks/04_position_sizing_optimization.ipynb` - Uses old backtest_engine
- ⚠️ `notebooks/03_backtest_momentum.ipynb` - Uses old backtest_engine

---

## Migration Guide

### For Old PaperTrader Users

**Old Code:**
```python
from core.paper_trader import PaperTrader

trader = PaperTrader(initial_cash=100000)
result = trader.simulate(df, position_col='Position')
```

**New Code:**
```python
from core.paper_trading_engine import PaperTradingEngine
from core.portfolio_manager import PortfolioConfig

config = PortfolioConfig(initial_capital=100000, ...)
engine = PaperTradingEngine(config=config, ...)
engine.initialize(prices_dict, signals_dict)
status = engine.get_portfolio_status(prices_dict)
```

### For Old Backtest Engine Users

**Old Code:**
```python
from core.backtest_engine import run_walk_forward

results = run_walk_forward(
    df=data,
    signal_class=MomentumSignal,
    params={'lookback': 120}
)
```

**New Code:**
```python
from core.portfolio_manager import run_multi_asset_backtest
from core.multi_asset_signal import SingleAssetWrapper

signal_gen = MomentumSignalV2(lookback=120)
wrapper = SingleAssetWrapper(signal_gen)
signals = wrapper.generate(prices_dict)

result, equity, trades = run_multi_asset_backtest(
    signals_dict=signals,
    prices_dict=prices_dict,
    config=config
)
```

---

## Files That Need Updating

### High Priority

1. **`scripts/run_daily.py`**
   - Still imports `paper_trader` and `backtest_engine`
   - Should be rewritten to use `PaperTradingEngine`
   - See `tests/test_paper_trading_engine.py` for daily workflow

2. **`utils/load_params.py`**
   - Imports old `paper_trader`
   - May need refactoring

### Medium Priority

3. **`notebooks/03_backtest_momentum.ipynb`**
   - Uses `run_walk_forward` from old engine
   - Should migrate to `run_multi_asset_backtest`

4. **`notebooks/04_position_sizing_optimization.ipynb`**
   - Uses `run_walk_forward` and old `PaperTrader`
   - Should migrate to new architecture

5. **`core/optimizer.py`**
   - May depend on old structures
   - Should be updated to work with new portfolio_manager

---

## Archive Structure

```
archive/
├── old_core/
│   ├── paper_trader.py              # Old single-asset paper trader
│   ├── backtest_engine.py           # Old walk-forward engine
│   └── risk_manager_old.py          # Old risk manager stub
│
├── old_scripts/
│   ├── test_new_structure.py        # Structure validation tests
│   ├── test_quick_fixes.py          # Ad-hoc fix tests
│   ├── test_diagnostics.py          # Old diagnostics
│   ├── test_stop_loss.py            # Stop-loss with old trader
│   ├── test_optimizer.py            # Old optimizer tests
│   ├── test_perfold_optimization.py # Per-fold optimization
│   ├── test_position_sizing.py      # Position sizing with old engine
│   └── sanity_check_signal.py       # Signal checks with old trader
│
└── old_structure/                    # Previous archive from restructure
    ├── backtest/
    ├── live/
    └── signals/
```

---

## Recommendations

### Immediate Actions Needed

1. ✅ Archive completed - legacy files moved
2. ⚠️ Update `scripts/run_daily.py` to use `PaperTradingEngine`
3. ⚠️ Review `utils/load_params.py` for old imports
4. ⚠️ Add migration guide to main README

### Future Cleanup

1. **Notebooks 03-04**: Migrate to new architecture or mark as "legacy examples"
2. **core/optimizer.py**: Update to work with new portfolio_manager
3. **core/reporter.py**: Consider if functionality should merge with RiskDashboard
4. **scripts/run_daily.py**: Complete rewrite using PaperTradingEngine

### Testing Requirements

After updates:
1. Run `tests/test_paper_trading_engine.py` ✅
2. Run `scripts/test_risk_manager.py` ✅
3. Run `scripts/test_risk_integration.py` ✅
4. Validate notebooks 09-10 ✅
5. Test updated run_daily.py (TODO)

---

## Benefits of Cleanup

### Before Cleanup
- 15 files in core/
- 20 test scripts
- 3 risk manager implementations (confusing)
- 2 paper trading implementations
- Mixed old/new patterns

### After Cleanup
- 11 active files in core/ (27% reduction)
- 12 active scripts (40% reduction)
- 1 risk manager (clear)
- 1 paper trading engine (clear)
- Consistent modern patterns

### Impact
- ✅ Clearer codebase structure
- ✅ No confusion about which modules to use
- ✅ Easier onboarding for new developers
- ✅ Reduced maintenance burden
- ✅ Old code preserved in archive if needed
- ✅ Clear migration path documented

---

## Rollback Instructions

If archived files are needed:

```bash
# Restore a specific file
cp archive/old_core/paper_trader.py core/

# Restore all old core files
cp archive/old_core/* core/

# Restore all old scripts
cp archive/old_scripts/* scripts/
```

However, consider migrating to new architecture instead of rolling back.

---

## Summary

Successfully archived 11 redundant files:
- 3 core modules (old implementations)
- 8 test scripts (obsolete functionality)

All archived code is preserved and can be referenced if needed, but the codebase now has a clear, modern architecture focused on:
- `PaperTradingEngine` for paper trading
- `portfolio_manager` for backtesting
- `risk_manager` for risk controls
- Clean separation of concerns
- Production-ready workflows

Next steps: Update remaining scripts and notebooks to use new architecture.
