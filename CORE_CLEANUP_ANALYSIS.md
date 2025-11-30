# Core Folder Cleanup Analysis

## Current Structure

### `/core/` (Root Level)
```
core/
├── __init__.py
├── BACKTEST_README.md
├── backtest.py                    # ⚠️ OLD - walk-forward optimizer
├── benchmark.py                   # ✅ KEEP - benchmark utilities
├── multi_asset_loader.py          # ✅ KEEP - data loading (used by notebooks)
├── multi_asset_signal.py          # ✅ KEEP - signal generation wrapper
├── multi_strategy_reporter.py     # ⚠️ REVIEW - may be superseded by reporter.py
├── multi_strategy_signal.py       # ✅ KEEP - multi-strategy signal generation
├── optimizer.py                   # ⚠️ REVIEW - parameter optimization
├── paper_trading_engine.py        # ✅ KEEP - live trading engine
├── portfolio_manager.py           # ⚠️ DUPLICATE - OLD VERSION
├── position_sizers.py             # ⚠️ DUPLICATE - OLD VERSION  
├── reporter.py                    # ✅ KEEP - used by notebooks
├── risk_dashboard.py              # ✅ KEEP - used by notebooks
├── risk_manager.py                # ⚠️ DUPLICATE - OLD VERSION
├── strategy_config.py             # ⚠️ REVIEW - may be obsolete
└── strategy_selector.py           # ⚠️ REVIEW - may be obsolete
```

### `/core/portfolio/` (New V2 Architecture)
```
core/portfolio/
├── __init__.py
├── backtest_result.py             # ✅ NEW - backtest result container
├── execution_engine.py            # ✅ NEW - transaction cost simulation
├── portfolio.py                   # ✅ NEW - portfolio state tracking
├── portfolio_manager_v2.py        # ✅ NEW - orchestration layer
├── position_sizers.py             # ✅ NEW - 5 position sizing strategies
└── risk_manager.py                # ✅ NEW - risk enforcement with DI
```

---

## Analysis: Duplicates & Obsolete Code

### 🔴 DUPLICATES (Remove from `/core/`)

#### 1. `core/portfolio_manager.py` → **REMOVE**
- **Old version**: 618 lines, uses static allocation
- **New version**: `core/portfolio/portfolio_manager_v2.py` (346 lines)
- **Why remove**: V2 is cleaner, modular, supports multiple position sizers
- **What uses old version**: 
  - `core/backtest.py` (walk-forward optimizer - also obsolete)
  - Archived notebooks (already in archive/)
- **Migration**: All active notebooks use V2

#### 2. `core/position_sizers.py` → **REMOVE**
- **Old version**: 266 lines, only VolatilityTargeting class
- **New version**: `core/portfolio/position_sizers.py` (446 lines, 5 strategies)
- **Why remove**: New version has 5 position sizing strategies vs 1
- **What uses old version**: None (checked grep results)
- **Migration**: Complete - notebooks use new version

#### 3. `core/risk_manager.py` → **REMOVE** 
- **Old version**: 329 lines, does position sizing internally
- **New version**: `core/portfolio/risk_manager.py` (478 lines)
- **Why remove**: New version uses dependency injection for position sizers
- **What uses old version**:
  - `core/portfolio_manager.py` (which we're removing)
  - `core/paper_trading_engine.py` (needs update - see below)
  - Archived notebooks
- **Migration**: Need to update `paper_trading_engine.py`

---

### ⚠️ NEEDS REVIEW (May Be Obsolete)

#### 4. `core/backtest.py` → **LIKELY REMOVE**
- **Purpose**: Walk-forward optimization framework
- **Issues**: 
  - Imports old `PortfolioManager` (not V2)
  - 500+ lines, complex
  - Notebook workflow is preferred
- **Decision**: Archive if no longer needed for walk-forward

#### 5. `core/multi_strategy_reporter.py` → **REVIEW**
- **Purpose**: Multi-strategy HTML reporting
- **Potential duplicate**: `core/reporter.py` (also does multi-strategy)
- **Check**: Do they have different purposes or is one obsolete?

#### 6. `core/optimizer.py` → **REVIEW**
- **Purpose**: Parameter optimization (grid search, optuna)
- **Question**: Still used? Or replaced by walk-forward in notebooks?

#### 7. `core/strategy_config.py` → **REVIEW**
- **Purpose**: Strategy configuration management
- **Question**: Still needed or replaced by direct config in notebooks?

#### 8. `core/strategy_selector.py` → **REVIEW**
- **Purpose**: Strategy selection logic
- **Question**: Still used in walk-forward or obsolete?

---

### ✅ KEEP (Essential & Used)

#### `core/multi_asset_loader.py`
- **Used by**: All notebooks, `backtest_with_risk_controls.ipynb`
- **Purpose**: Load futures data from CSV or yfinance
- **Status**: Essential

#### `core/multi_asset_signal.py`
- **Used by**: Notebooks, paper trading
- **Purpose**: Wrapper for single-asset signals in multi-asset context
- **Status**: Essential

#### `core/multi_strategy_signal.py`
- **Used by**: Notebooks for ensemble strategies
- **Purpose**: Combine multiple strategies
- **Status**: Essential

#### `core/reporter.py`
- **Used by**: `backtest_with_risk_controls.ipynb`
- **Purpose**: Generate HTML reports with Plotly charts
- **Status**: Essential

#### `core/risk_dashboard.py`
- **Used by**: `backtest_with_risk_controls.ipynb`
- **Purpose**: Risk analytics and VaR/CVaR dashboards
- **Status**: Essential

#### `core/paper_trading_engine.py`
- **Used by**: Live trading workflows
- **Purpose**: State persistence, live order execution
- **Status**: Essential (but needs update for new RiskManager)

#### `core/benchmark.py`
- **Purpose**: Benchmark comparison utilities
- **Status**: Keep (may be used by reporter/dashboard)

---

## 🔧 Required Updates

### 1. Update `paper_trading_engine.py`
**Issue**: Imports old `RiskManager` from `core.risk_manager`

**Fix**:
```python
# OLD
from core.risk_manager import RiskManager, RiskConfig
from core.portfolio_manager import PortfolioConfig

# NEW
from core.portfolio.risk_manager import RiskManager, RiskConfig
from core.portfolio.portfolio_manager_v2 import PortfolioManagerV2
```

**Also update**: PortfolioConfig references → adapt to V2 architecture

---

## 📋 Cleanup Action Plan

### Phase 1: Archive Duplicates (Safe - Have Backups)
```bash
# Create archive folder
mkdir -p archive/cleanup_2025_11_30/old_core_duplicates

# Move duplicates
mv core/portfolio_manager.py archive/cleanup_2025_11_30/old_core_duplicates/
mv core/position_sizers.py archive/cleanup_2025_11_30/old_core_duplicates/
mv core/risk_manager.py archive/cleanup_2025_11_30/old_core_duplicates/
```

### Phase 2: Review & Decide
1. **Check if `backtest.py` is still needed** for walk-forward
   - If yes: Update to use V2 portfolio manager
   - If no: Archive it

2. **Compare `reporter.py` vs `multi_strategy_reporter.py`**
   - Determine if one supersedes the other
   - Keep the better one, archive the other

3. **Check `optimizer.py`, `strategy_config.py`, `strategy_selector.py`**
   - Are they used anywhere?
   - If unused: Archive them

### Phase 3: Update Dependencies
1. **Update `paper_trading_engine.py`** 
   - Import from `core.portfolio.*`
   - Adapt to new RiskManager API

2. **Test Everything**
   - Run `notebooks/backtest_with_risk_controls.ipynb`
   - Test paper trading engine
   - Verify reports generate correctly

---

## Summary

### Can Remove Immediately (Duplicates)
- ✅ `core/portfolio_manager.py` → `core/portfolio/portfolio_manager_v2.py`
- ✅ `core/position_sizers.py` → `core/portfolio/position_sizers.py`
- ✅ `core/risk_manager.py` → `core/portfolio/risk_manager.py`

### Need Investigation
- ⚠️ `core/backtest.py` - Check if walk-forward is still used
- ⚠️ `core/multi_strategy_reporter.py` - Compare with `reporter.py`
- ⚠️ `core/optimizer.py` - Check if still used
- ⚠️ `core/strategy_config.py` - Check if still used
- ⚠️ `core/strategy_selector.py` - Check if still used

### Keep & Maintain
- ✅ `core/multi_asset_loader.py`
- ✅ `core/multi_asset_signal.py`
- ✅ `core/multi_strategy_signal.py`
- ✅ `core/reporter.py`
- ✅ `core/risk_dashboard.py`
- ✅ `core/paper_trading_engine.py` (after update)
- ✅ `core/benchmark.py`

### Required Work
1. Update `paper_trading_engine.py` to use `core.portfolio.*` imports
2. Test paper trading engine with new RiskManager
3. Investigate 5 "needs review" files
4. Archive confirmed obsolete files

---

**Next Steps**: Shall I proceed with Phase 1 (archive duplicates) and investigate the "needs review" files?
