# Repository Cleanup - November 29, 2025

## Purpose
Clean up obsolete code, test scripts, and duplicate modules to streamline the repository structure.

## What Was Archived

### 1. **Obsolete Modules** (Replaced by `core/portfolio/`)
- `backtest/` - Old backtest module (now using `PortfolioManagerV2`)
- `analysis/` - Old reporting code (replaced by `core/reporter.py` and `core/risk_dashboard.py`)
- `features/` - Empty folder (never used)

### 2. **Old Core Files**
- `core/risk_dashboard_old.py` - Replaced by `core/risk_dashboard.py`

### 3. **Test & Debug Scripts** (`old_test_scripts/`)
All moved from `/scripts/`:
- `test_multi_strategy.py`
- `test_paper_trading_engine.py`
- `test_gc_load.py`
- `test_multi_asset_simple.py`
- `test_portfolio_allocation.py`
- `test_risk_manager.py`
- `test_risk_integration.py`
- `test_reporter.py`
- `debug_portfolio.py`
- `compare_backtest_performance.py`
- `find_best_trial.py`
- `validate_futures_data.py`

### 4. **Old Data Files**
- `data/paper_trading_state.pkl`
- `data/paper_trading_state_test.pkl`

### 5. **Old Notebooks** (`old_notebooks/`)
- `portfolio_refactor_comparison.ipynb` - Comparison during refactor (no longer needed)
- `test_new_signals.ipynb` - Signal testing notebook
- `testing/` - Old testing directory
- `step_through_guide/` - Old tutorial notebooks

## Current Active Structure

### Core Modules (Production)
```
core/
├── portfolio/
│   ├── portfolio_manager_v2.py  ✅ Main backtest orchestrator
│   ├── risk_manager.py          ✅ Risk controls
│   ├── execution_engine.py      ✅ Trade execution
│   ├── portfolio.py             ✅ State tracking
│   └── backtest_result.py       ✅ Results container
├── reporter.py                  ✅ HTML report generation
├── risk_dashboard.py            ✅ Risk visualization
├── benchmark.py                 ✅ Benchmark comparison
├── multi_asset_loader.py        ✅ Data loading
└── strategy_config.py           ✅ Strategy configuration
```

### Signal Generators (Production)
```
signals/
├── base.py                      ✅ Base class
├── momentum.py                  ✅ Momentum strategies
├── mean_reversion.py            ✅ Mean reversion
├── trend_following_long_short.py ✅ Trend following
├── ensemble.py                  ✅ Multi-strategy ensemble
└── hybrid_adaptive.py           ✅ Adaptive strategies
```

### Active Notebooks
```
notebooks/
├── backtest_with_risk_controls.ipynb    ✅ Main production notebook
├── multi_strategy_portfolio.ipynb       ✅ Multi-strategy demos
├── multi_strategy_with_ensemble.ipynb   ✅ Ensemble strategies
├── risk_controls_demo.ipynb             ✅ Risk demo
└── walk_forward_validation.ipynb        ✅ Walk-forward testing
```

### Scripts (Production)
```
scripts/
└── run_daily.py                 ✅ Live trading execution
```

## Why These Were Archived

1. **backtest/** - Superseded by `core/portfolio/portfolio_manager_v2.py` architecture
2. **analysis/** - Replaced by comprehensive `core/reporter.py` and `core/risk_dashboard.py`
3. **Test scripts** - One-off testing during development; all functionality now in notebooks
4. **Old notebooks** - Comparison/testing notebooks from previous refactors
5. **Pickle files** - Old paper trading state files (not used in production)

## Impact Analysis

✅ **No Breaking Changes** - All archived code was already replaced or unused

### What Still Works:
- All signal generators
- Portfolio backtesting (`PortfolioManagerV2`)
- HTML report generation
- Risk dashboards
- Multi-strategy orchestration
- Walk-forward validation
- Benchmark comparison

### Dependencies Removed:
- None - all archived code was already deprecated

## Recovery

If any archived code is needed, it can be restored from this directory. All files are preserved with their original structure.

## Next Steps

Consider removing in future cleanups:
1. `Dataset/` folder (empty or merge with `/data/`)
2. Old log files in `logs/` (keep recent only)
3. Old reports in `reports/` (archive older than 30 days)
4. Review `utils/` folder for unused utility functions
5. Check if `core/backtest.py`, `core/portfolio_manager.py` (old v1) can be archived
