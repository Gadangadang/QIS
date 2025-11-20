# Archived Code - Old Structure

**Archived Date:** November 20, 2025

## What's Archived

This folder contains the old codebase structure before the major restructuring.

### Folders Archived

1. **`backtest/`** - Old backtesting scripts
   - `runner.py` - 500+ line CLI with 15+ flags (replaced by notebooks)
   - `run_walkforward.py` - Standalone walk-forward runner (replaced by notebooks)
   - `walk_forward.py` - Alternative WF implementation (not needed)
   - `run_diagnostics.py` - Manual diagnostics script (replaced by BacktestReport)
   - `backtest_engine.py` - Original version (now in `core/`)
   - `sanity_check_signal.py` - Original version (now in `scripts/`)
   - `test_stop_loss.py` - Original version (now in `scripts/`)

2. **`live/`** - Old live trading scripts
   - `paper_trader.py` - Original version (now in `core/`)
   - `run_daily.py` - Original version (now in `scripts/`)

3. **`utils/`** - Old utility modules
   - `metrics.py` - Original version (now in `analysis/`)
   - `logger.py` - Daily logging utility (may still be useful)

## Why Archived

- **CLI workflow replaced** by Jupyter notebook workflow
- **Files moved** to new logical structure (`core/`, `analysis/`, `scripts/`)
- **Enhanced functionality** in new versions with auto-report generation
- **Better organization** for maintainability and scalability

## New Structure

```
QuantTrading/
├── core/           # Execution engine
├── analysis/       # Reporting & metrics
├── signals/        # Signal models
├── notebooks/      # Research notebooks
├── scripts/        # Utility scripts
└── configs/        # Configuration files
```

## Recovery

If you need to recover any of these files:

```bash
# View archived files
ls -la archive/old_structure/

# Copy specific file back
cp archive/old_structure/backtest/runner.py ./

# Or restore entire folder
cp -r archive/old_structure/backtest ./
```

## Safe to Delete?

After confirming the new structure works for 1-2 weeks, this archive folder can be safely deleted:

```bash
rm -rf archive/
```

## Testing Status

✅ All tests passed after archiving old code
✅ New structure verified working
✅ HTML reports generating correctly
✅ All imports resolved properly

---

**Note:** Keep this archive for at least 1 week to ensure the new structure works completely in all use cases.
