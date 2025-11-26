# Backtest Module

**Location:** `core/backtest.py`

## Overview

Central module for all backtesting functionality including single-period backtests and walk-forward optimization.

## What Moved Here

Previously, backtest functions lived in `portfolio_manager.py`. They have been moved here for better organization:

- `run_multi_asset_backtest()` - Main backtest entry point
- `_run_backtest()` - Core backtest implementation  
- `WalkForwardEngine` - NEW: Production walk-forward optimization class

## Backwards Compatibility

**Old imports still work:**
```python
from core.portfolio_manager import run_multi_asset_backtest
```

This is maintained through compatibility imports in `portfolio_manager.py`.

**New recommended imports:**
```python
from core.backtest import run_multi_asset_backtest, WalkForwardEngine
```

## Classes & Functions

### `run_multi_asset_backtest()`
Main entry point for running a backtest on multiple assets.

**Args:**
- `signals_dict`: Dict[ticker → DataFrame with 'Signal' column]
- `prices_dict`: Dict[ticker → DataFrame with OHLC data]
- `config`: PortfolioConfig object
- `return_pm`: If True, return PortfolioManager; else BacktestResult

**Returns:**
- `(result, equity_curve, trades)` tuple

**Example:**
```python
from core.backtest import run_multi_asset_backtest
from core.portfolio_manager import PortfolioConfig

result, equity_curve, trades = run_multi_asset_backtest(
    signals, prices, config
)
metrics = result.calculate_metrics()
print(f"Sharpe: {metrics['Sharpe Ratio']:.2f}")
```

### `WalkForwardEngine`
Production-ready walk-forward optimization engine.

**Features:**
- ✅ Rolling window optimization
- ✅ Parameter grid search
- ✅ Out-of-sample validation
- ✅ Aggregate OOS metrics
- ✅ Best parameters from latest window

**Example:**
```python
from core.backtest import WalkForwardEngine
from signals.momentum import MomentumSignalV2

engine = WalkForwardEngine(
    signal_class=MomentumSignalV2,
    param_grid={
        'lookback': [60, 90, 120],
        'sma_filter': [150, 200, 250]
    },
    tickers=['ES', 'GC', 'NQ'],
    start_date='2010-01-01',
    end_date='2024-12-31',
    train_years=3,
    test_years=1
)

results = engine.run()

print(f"Best params: {results['best_params']}")
print(f"Avg OOS Sharpe: {results['oos_sharpe_mean']:.2f}")
print(f"Windows tested: {len(results['windows'])}")
```

## Integration with Notebooks

### Step-Through Guide
- `05_walk_forward_framework.ipynb` - Tutorial explaining walk-forward concepts
- `04_full_system_integration.ipynb` - Can now toggle walk-forward on/off

### Usage Pattern
```python
# Toggle walk-forward optimization
USE_WALK_FORWARD = True

if USE_WALK_FORWARD:
    # Optimize parameters
    engine = WalkForwardEngine(...)
    wf_results = engine.run()
    best_params = wf_results['best_params']
    
    # Use optimized params
    signal_gen = MomentumSignalV2(**best_params)
else:
    # Use fixed parameters
    signal_gen = MomentumSignalV2(lookback=120, sma_filter=200)

# Rest is identical
signals = SingleAssetWrapper(signal_gen).generate(prices)
result = run_multi_asset_backtest(signals, prices, config)
```

## Walk-Forward Results Structure

```python
{
    'windows': [
        {
            'window': 1,
            'train_start': '2010-01-01',
            'train_end': '2013-01-01',
            'test_start': '2013-01-02',
            'test_end': '2014-01-02',
            'best_params': {'lookback': 60, 'sma_filter': 150},
            'is_sharpe': 0.86,
            'oos_sharpe': 1.91,
            'oos_return': 0.1204,
            ...
        },
        ...
    ],
    'best_params': {'lookback': 90, 'sma_filter': 200},  # From latest window
    'oos_sharpe_mean': 1.23,
    'oos_return_mean': 0.0842,
    'summary_df': DataFrame(...)  # All windows in tabular format
}
```

## Testing

Run the module directly to execute self-tests:

```bash
cd /Users/Sakarias/QuantTrading
python -m core.backtest
```

## Performance Considerations

**Walk-forward optimization can be slow:**
- 4 windows × 12 param combos × 3 assets = 48 backtests
- Each backtest processes 500-750 days of data
- Typical runtime: 2-5 minutes for full optimization

**Optimization tips:**
- Start with small parameter grids (3-4 combinations)
- Use shorter train/test periods for prototyping
- Consider parallel processing for production (future enhancement)

## Future Enhancements

- [ ] Parallel parameter optimization
- [ ] Anchored walk-forward (vs rolling)
- [ ] Custom optimization metrics
- [ ] Multi-strategy walk-forward
- [ ] Walk-forward visualization tools
