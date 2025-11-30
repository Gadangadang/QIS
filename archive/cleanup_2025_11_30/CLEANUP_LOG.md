# Core Folder Cleanup - November 30, 2025

## Summary
Archived duplicate and obsolete files from `/core/` to clean up architecture after V2 portfolio manager implementation.

---

## Files Archived

### ✅ Duplicates (Replaced by core/portfolio/* modules)

1. **`core/portfolio_manager.py`** → `archive/cleanup_2025_11_30/old_core_duplicates/`
   - **Status**: REPLACED by `core/portfolio/portfolio_manager_v2.py`
   - **Reason**: V2 is cleaner, modular, supports pluggable position sizers
   - **Used by**: Only archived notebooks and obsolete scripts
   - **Size**: 618 lines → 346 lines in V2

2. **`core/position_sizers.py`** → `archive/cleanup_2025_11_30/old_core_duplicates/`
   - **Status**: REPLACED by `core/portfolio/position_sizers.py`
   - **Reason**: New version has 5 sizing strategies vs 1 (VolatilityTargeting)
   - **Used by**: Nothing (verified with grep)
   - **Size**: 266 lines → 446 lines in V2 (5x strategies)

3. **`core/risk_manager.py`** → `archive/cleanup_2025_11_30/old_core_duplicates/`
   - **Status**: REPLACED by `core/portfolio/risk_manager.py`
   - **Reason**: New version uses dependency injection for position sizers
   - **Used by**: Old portfolio_manager.py (also archived)
   - **Size**: 329 lines → 478 lines in V2 (better architecture)

---

### ✅ Obsolete Modules (Not Used in Active Workflows)

4. **`core/backtest.py`** → `archive/cleanup_2025_11_30/old_core_duplicates/`
   - **Status**: OBSOLETE - only used in archived notebooks
   - **Purpose**: Walk-forward optimization framework
   - **Reason**: 
     - Imports old PortfolioManager (not V2)
     - 500+ lines, complex
     - Notebook workflow is now preferred
   - **Used by**: 
     - `core/BACKTEST_README.md` (documentation only)
     - Archived notebook `05_walk_forward_framework.ipynb`

5. **`core/optimizer.py`** → `archive/cleanup_2025_11_30/old_core_duplicates/`
   - **Status**: OBSOLETE - only used in archived files
   - **Purpose**: Parameter optimization (grid search, optuna)
   - **Used by**:
     - `readmes/OPTIMIZER_USAGE.md` (old documentation)
     - `archive/old_scripts/test_optimizer.py`
     - Archived notebook `04_position_sizing_optimization.ipynb`

6. **`core/strategy_config.py`** → `archive/cleanup_2025_11_30/old_core_duplicates/`
   - **Status**: OBSOLETE - not used anywhere
   - **Purpose**: Strategy configuration management
   - **Used by**: Only `readmes/ARCHITECTURE.md` (old documentation)

7. **`core/strategy_selector.py`** → `archive/cleanup_2025_11_30/old_core_duplicates/`
   - **Status**: OBSOLETE - not used anywhere
   - **Purpose**: Strategy selection logic for walk-forward
   - **Used by**: Only `readmes/ARCHITECTURE.md` (old documentation)

---

### ⚠️ Kept But Marked Deprecated

8. **`core/paper_trading_engine.py`** → KEPT (with deprecation notice)
   - **Status**: DEPRECATED but not archived
   - **Purpose**: Live trading state persistence and incremental updates
   - **Reason for keeping**: 
     - Provides unique functionality (state persistence, daily workflow)
     - Not duplicated by V2 architecture
     - May be needed for future live trading
     - Would be significant work to rebuild
   - **Action taken**: 
     - Added deprecation warning in docstring
     - Imports point to archived modules
     - Marked as "needs refactoring for V2"
   - **Used by**: Only archived notebook `10_paper_trading_with_engine.ipynb`

---

## Files Kept (Active Use)

### ✅ Essential Modules
- **`core/multi_asset_loader.py`** - Data loading (used by all active notebooks)
- **`core/multi_asset_signal.py`** - Signal wrapper (used by notebooks)
- **`core/multi_strategy_signal.py`** - Multi-strategy signals (used by notebooks)
- **`core/reporter.py`** - HTML report generation (used by `backtest_with_risk_controls.ipynb`)
- **`core/risk_dashboard.py`** - Risk dashboards (used by `backtest_with_risk_controls.ipynb`)
- **`core/benchmark.py`** - Benchmark utilities (used by reporter/dashboard)

### ⚠️ Need Investigation
- **`core/multi_strategy_reporter.py`** - May duplicate `reporter.py`, needs comparison

---

## Active Notebooks Using V2 Architecture

✅ **`notebooks/backtest_with_risk_controls.ipynb`** - PRIMARY WORKFLOW
- Uses: `core.portfolio.portfolio_manager_v2.PortfolioManagerV2`
- Uses: `core.portfolio.position_sizers.*` (all 5 strategies)
- Uses: `core.reporter.Reporter`
- Uses: `core.risk_dashboard.RiskDashboard`
- Status: ✅ Fully migrated to V2

Other active notebooks:
- `notebooks/multi_strategy_portfolio.ipynb` - Uses `multi_strategy_reporter`
- `notebooks/multi_strategy_with_ensemble.ipynb` - Uses `multi_strategy_reporter`  
- `notebooks/risk_controls_demo.ipynb`
- `notebooks/walk_forward_validation.ipynb`

---

## Migration Notes

### For Old Code Using Archived Modules:

**Old imports:**
```python
from core.portfolio_manager import PortfolioManager, PortfolioConfig
from core.risk_manager import RiskManager, RiskConfig
from core.position_sizers import VolatilityTargeting
```

**New imports:**
```python
from core.portfolio.portfolio_manager_v2 import PortfolioManagerV2
from core.portfolio.risk_manager import RiskManager, RiskConfig
from core.portfolio.position_sizers import (
    FixedFractionalSizer,
    KellySizer,
    ATRSizer,
    VolatilityScaledSizer,
    RiskParitySizer
)
```

### Key Architecture Changes:

1. **Position Sizers**: Now injected via dependency injection
   ```python
   sizer = ATRSizer(risk_per_trade=0.02, max_position_pct=0.25)
   pm = PortfolioManagerV2(initial_capital=100000, position_sizer=sizer)
   ```

2. **RiskManager**: Accepts position_sizer parameter
   ```python
   risk_mgr = RiskManager(config, position_sizer=sizer)
   ```

3. **No PortfolioConfig**: V2 uses direct parameters
   ```python
   # OLD
   config = PortfolioConfig(initial_capital=100000, max_position_size=0.25)
   pm = PortfolioManager(config)
   
   # NEW
   pm = PortfolioManagerV2(initial_capital=100000, max_position_size=0.25)
   ```

---

## Verification

### Grep Tests Performed:
```bash
# Check for imports of old modules
grep -r "from core.portfolio_manager import" --include="*.py" --include="*.ipynb"
grep -r "from core.risk_manager import" --include="*.py" --include="*.ipynb"
grep -r "from core.position_sizers import" --include="*.py" --include="*.ipynb"
grep -r "from core.backtest import" --include="*.py" --include="*.ipynb"
grep -r "from core.optimizer import" --include="*.py" --include="*.ipynb"
```

### Results:
- ✅ No active notebooks use old modules
- ✅ Only archived notebooks and documentation reference old modules
- ✅ `backtest_with_risk_controls.ipynb` uses V2 exclusively

---

## TODO: Future Work

1. **Compare `reporter.py` vs `multi_strategy_reporter.py`**
   - Determine if one supersedes the other
   - Archive the obsolete one

2. **Refactor `paper_trading_engine.py`**
   - Update to use V2 portfolio manager
   - Adapt to new position sizer architecture
   - Test with current notebooks

3. **Review other active notebooks**
   - Check if they need migration to V2
   - Update if using old modules

---

## File Locations

### Before Cleanup:
```
core/
├── backtest.py
├── optimizer.py
├── portfolio_manager.py
├── position_sizers.py
├── risk_manager.py
├── strategy_config.py
├── strategy_selector.py
└── paper_trading_engine.py (marked deprecated)
```

### After Cleanup:
```
core/
├── paper_trading_engine.py (deprecated, needs refactoring)
└── portfolio/
    ├── backtest_result.py
    ├── execution_engine.py
    ├── portfolio.py
    ├── portfolio_manager_v2.py
    ├── position_sizers.py
    └── risk_manager.py

archive/cleanup_2025_11_30/old_core_duplicates/
├── backtest.py
├── optimizer.py
├── portfolio_manager.py
├── position_sizers.py
├── risk_manager.py
├── strategy_config.py
└── strategy_selector.py
```

---

**Cleanup Date**: November 30, 2025  
**Performed By**: GitHub Copilot + User  
**Reason**: Architecture consolidation after V2 implementation  
**Status**: ✅ Complete - Ready to commit
