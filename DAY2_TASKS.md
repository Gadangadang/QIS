# Day 2 Tasks: Performance & Reporting

**Date:** November 24, 2025  
**Goal:** 10-year, 3-asset backtest in <5 seconds + HTML reporting integration

---

## 🎯 Success Criteria

- [ ] Baseline performance documented (current timing)
- [ ] Portfolio calculations vectorized (no loops)
- [ ] <5 seconds for 10-year, 3-asset backtest
- [ ] Progress bars working for long operations
- [ ] HTML reports generated from multi-asset portfolios
- [ ] Reports include: equity curves, allocation, correlations, metrics

---

## ⏰ Morning Tasks (9am - 12pm): Profiling & Setup

### Task 1: Create Profiling Script (30 min)

**File:** `scripts/profile_backtest.py`

```python
"""
Profile multi-asset portfolio backtest to identify bottlenecks.
"""
import cProfile
import pstats
import io
from pstats import SortKey

from core.portfolio_manager import PortfolioManager
from core.multi_asset_loader import load_futures_data
from core.multi_strategy_signal import StrategyConfig

def run_benchmark():
    """Run standard 10-year, 3-asset backtest."""
    # Load data (2014-2024)
    prices = load_futures_data(['ES', 'NQ', 'GC'], 
                                start_date='2014-01-01',
                                end_date='2024-12-31')
    
    # Configure strategies
    config = (StrategyConfig()
              .add_momentum('ES', lookback=120, entry_threshold=0.02)
              .add_momentum('NQ', lookback=90, entry_threshold=0.03)
              .add_mean_reversion('GC', window=50, entry_z=2.0)
              .build())
    
    signals = config.generate(prices)
    
    # Run portfolio backtest
    pm = PortfolioManager(
        initial_capital=100000,
        risk_per_trade=0.02,
        max_position_size=0.20,
        rebalance_frequency='monthly'
    )
    
    equity_curve, trades = pm.run_backtest(signals, prices)
    return equity_curve, trades

if __name__ == '__main__':
    print("Starting performance profiling...")
    
    # Profile the benchmark
    pr = cProfile.Profile()
    pr.enable()
    
    import time
    start = time.time()
    equity, trades = run_benchmark()
    end = time.time()
    
    pr.disable()
    
    # Print timing
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {end - start:.2f} seconds")
    print(f"{'='*60}\n")
    
    # Print top 20 slowest functions
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(20)
    print(s.getvalue())
    
    # Save full report
    with open('logs/profile_report.txt', 'w') as f:
        ps = pstats.Stats(pr, stream=f).sort_stats(SortKey.CUMULATIVE)
        ps.print_stats()
    
    print(f"\nFull report saved to: logs/profile_report.txt")
    print(f"Number of trades: {len(trades)}")
    print(f"Final equity: ${equity.iloc[-1]:,.2f}")
```

**Action Items:**
- [ ] Create the script
- [ ] Run it and document baseline timing
- [ ] Identify top 5 slowest functions
- [ ] Save results to `logs/baseline_performance.txt`

---

### Task 2: Add Timing Decorator (20 min)

**File:** `utils/logger.py`

Add this decorator:

```python
import time
import functools
import logging

def timing_decorator(func):
    """
    Decorator to measure and log function execution time.
    
    Usage:
        @timing_decorator
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        elapsed = end - start
        logger = logging.getLogger(__name__)
        logger.info(f"{func.__name__} took {elapsed:.3f} seconds")
        
        return result
    return wrapper
```

**Action Items:**
- [ ] Add decorator to `utils/logger.py`
- [ ] Apply to key functions in `portfolio_manager.py`:
  - `run_backtest()`
  - `calculate_positions()`
  - `apply_rebalancing()`
  - `calculate_equity_curve()`
- [ ] Test that timing logs appear

---

### Task 3: Document Baseline Performance (10 min)

**File:** `logs/baseline_performance.txt`

Create document with:
- Total execution time
- Top 5 slowest functions with times
- Memory usage
- Number of data points
- Number of trades generated

**Template:**
```
BASELINE PERFORMANCE METRICS
============================
Date: 2025-11-24
Test: 10-year, 3-asset portfolio (ES, NQ, GC)

TIMING:
-------
Total execution time: X.XX seconds
Data loading: X.XX seconds
Signal generation: X.XX seconds
Portfolio calculations: X.XX seconds
Trade execution: X.XX seconds

TOP 5 BOTTLENECKS:
------------------
1. function_name: X.XX seconds
2. function_name: X.XX seconds
3. function_name: X.XX seconds
4. function_name: X.XX seconds
5. function_name: X.XX seconds

DATA VOLUME:
------------
Total data points: X,XXX
Date range: YYYY-MM-DD to YYYY-MM-DD
Number of assets: 3
Total trades: XXX

TARGET: <5 seconds
```

---

## 🚀 Afternoon Tasks (1pm - 5pm): Vectorization

### Task 4: Vectorize Portfolio Manager (2-3 hours)

**File:** `core/portfolio_manager.py`

#### Step 4.1: Identify All Loops

Search for these patterns:
```python
for date in dates:
for i in range(len(data)):
for idx, row in df.iterrows():
```

Document each loop location and what it does.

#### Step 4.2: Vectorize Position Calculations

**Before:**
```python
positions = {}
for date in dates:
    for ticker in tickers:
        position[date][ticker] = calculate_position(date, ticker)
```

**After:**
```python
# Vectorized using pandas operations
positions = pd.DataFrame()
for ticker in tickers:
    risk_amount = self.capital * self.risk_per_trade
    position_size = risk_amount / prices[ticker]
    positions[ticker] = position_size * signals[ticker].shift(1)
```

#### Step 4.3: Vectorize Rebalancing

Use pandas `.resample()` for time-based rebalancing:
```python
# Monthly rebalancing dates
rebalance_dates = prices.index[prices.index.to_series()
                               .resample('M').last().index]

# Apply rebalancing logic vectorized
for rebal_date in rebalance_dates:
    # Vectorized rebalance calculations here
```

#### Step 4.4: Vectorize Equity Curve

```python
# Before: Loop through each date
for date in dates:
    equity[date] = equity[date-1] + pnl[date]

# After: Cumulative sum
daily_pnl = (positions.shift(1) * prices.pct_change()).sum(axis=1)
equity_curve = (1 + daily_pnl).cumprod() * initial_capital
```

**Action Items:**
- [ ] Document all loops found
- [ ] Vectorize position calculations
- [ ] Vectorize rebalancing logic
- [ ] Vectorize equity curve calculation
- [ ] Test that results match original (within 0.01%)
- [ ] Measure new execution time

---

### Task 5: Add Progress Bars (30 min)

**Install tqdm:**
```bash
pip install tqdm
```

**Update walk-forward optimizer:**
```python
from tqdm import tqdm

# In optimizer.py
for fold_num in tqdm(range(self.n_folds), desc="Walk-Forward Folds"):
    # Optimization logic
    pass

# For parameter grid search
from itertools import product
param_combinations = list(product(*param_grid.values()))

for params in tqdm(param_combinations, desc="Testing Parameters"):
    # Test each parameter set
    pass
```

**Action Items:**
- [ ] Install tqdm
- [ ] Add progress bars to walk-forward folds
- [ ] Add progress bars to parameter grid search
- [ ] Test that progress displays correctly

---

## 🎨 Evening Tasks (6pm - 9pm): Reporting Integration

### Task 6: Extend BacktestReport for Multi-Asset (1.5 hours)

**File:** `analysis/report.py`

Add new class method:

```python
class BacktestReport:
    # ... existing code ...
    
    @classmethod
    def from_multi_asset(cls, equity_curve, trades, prices, signals):
        """
        Create report from multi-asset portfolio results.
        
        Args:
            equity_curve: pd.Series with portfolio equity over time
            trades: pd.DataFrame with columns [date, ticker, pnl, ...]
            prices: pd.DataFrame with prices for each asset (columns = tickers)
            signals: pd.DataFrame with signals for each asset (columns = tickers)
        
        Returns:
            BacktestReport instance
        """
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Create results dict matching expected format
        results = {
            'equity_curve': equity_curve,
            'returns': returns,
            'trades': trades,
            'prices': prices,
            'signals': signals
        }
        
        return cls(results)
    
    def plot_asset_allocation(self):
        """
        Plot stacked area chart showing allocation to each asset over time.
        """
        # Calculate position values by asset
        # Create plotly stacked area chart
        pass
    
    def signal_correlation_heatmap(self):
        """
        Plot heatmap of signal correlations between assets.
        """
        if 'signals' not in self.results:
            return None
        
        signals = self.results['signals']
        corr = signals.corr()
        
        # Create seaborn heatmap
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, ax=ax)
        ax.set_title('Signal Correlation Matrix')
        return fig
    
    def performance_by_asset(self):
        """
        Calculate and display performance metrics by asset.
        """
        if 'trades' not in self.results or 'ticker' not in self.results['trades'].columns:
            return None
        
        trades = self.results['trades']
        
        # Group by ticker
        by_asset = trades.groupby('ticker').agg({
            'pnl': ['count', 'sum', 'mean'],
            # Add more aggregations
        })
        
        return by_asset
```

**Action Items:**
- [ ] Add `from_multi_asset()` class method
- [ ] Implement `plot_asset_allocation()`
- [ ] Implement `signal_correlation_heatmap()`
- [ ] Implement `performance_by_asset()`
- [ ] Test with sample multi-asset data

---

### Task 7: Update Portfolio Manager Output Format (30 min)

**File:** `core/portfolio_manager.py`

Ensure `run_backtest()` returns data in format compatible with `BacktestReport`:

```python
def run_backtest(self, signals, prices):
    """
    Returns:
        equity_curve: pd.Series with DatetimeIndex
        trades: pd.DataFrame with columns:
            - entry_date: datetime
            - exit_date: datetime  
            - ticker: str
            - entry_price: float
            - exit_price: float
            - shares: float
            - pnl: float
            - return_pct: float
    """
    # ... existing logic ...
    
    # Format trades DataFrame
    trades_df = pd.DataFrame(self.trades)
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    
    # Format equity curve
    equity_series = pd.Series(equity_curve, index=dates)
    
    return equity_series, trades_df
```

**Action Items:**
- [ ] Verify output format matches BacktestReport expectations
- [ ] Add datetime conversions
- [ ] Ensure all required columns present
- [ ] Test integration

---

### Task 8: Add Reporting to Notebook (45 min)

**File:** `notebooks/05_multi_asset_demo.ipynb`

Add new section after Section 14:

```markdown
## Section 15: Interactive HTML Reports

Generate comprehensive HTML reports from portfolio backtests.
```

```python
from analysis.report import BacktestReport

# Generate report
report = BacktestReport.from_multi_asset(
    equity_curve=equity_curve,
    trades=trades,
    prices=prices,
    signals=signals
)

# Display summary metrics
print("=== PERFORMANCE SUMMARY ===")
report.summary()

# Plot equity curve
fig = report.plot_equity()
fig.show()

# Asset allocation over time
fig = report.plot_asset_allocation()
fig.show()

# Signal correlation heatmap
fig = report.signal_correlation_heatmap()
fig.show() if fig else None

# Performance by asset
by_asset = report.performance_by_asset()
print("\n=== PERFORMANCE BY ASSET ===")
print(by_asset)

# Save HTML report
report.save_html('../logs/multi_asset_report.html')
print("\nReport saved to: logs/multi_asset_report.html")
```

**Action Items:**
- [ ] Add Section 15 to notebook
- [ ] Test all visualizations work
- [ ] Ensure HTML report generates
- [ ] Verify report opens in browser
- [ ] Add markdown explanations

---

## ✅ End of Day Checklist

- [ ] Baseline performance documented
- [ ] All loops in portfolio_manager.py identified
- [ ] Position calculations vectorized
- [ ] Rebalancing logic vectorized
- [ ] Equity curve calculation vectorized
- [ ] Tests pass (results match within 0.01%)
- [ ] New execution time: <5 seconds ✓
- [ ] Progress bars working
- [ ] BacktestReport extended for multi-asset
- [ ] Portfolio manager output format updated
- [ ] Notebook Section 15 added
- [ ] HTML reports generating successfully
- [ ] Code committed to git with message: "Day 2: Performance optimization and reporting integration"

---

## 📊 Performance Targets

| Metric | Baseline | Target | Actual |
|--------|----------|--------|--------|
| Total Time | ??? sec | <5 sec | ??? |
| Data Loading | ??? | <0.5 sec | ??? |
| Signal Gen | ??? | <1 sec | ??? |
| Portfolio Calc | ??? | <2 sec | ??? |
| Trade Exec | ??? | <1 sec | ??? |

---

## 🔥 Troubleshooting

**If vectorization breaks results:**
- Compare old vs new equity curves (should match within 0.01%)
- Check for off-by-one errors with `.shift(1)`
- Verify date alignment across DataFrames
- Use `.loc[]` instead of chained indexing

**If progress bars don't show:**
- Ensure tqdm is installed: `pip install tqdm`
- Use `tqdm.notebook` version in Jupyter
- Check that loops have meaningful iteration counts

**If HTML reports fail:**
- Verify plotly is installed: `pip install plotly`
- Check data format matches expectations
- Ensure all required columns present in trades DataFrame
- Look for NaN values in calculations

---

## 💡 Optimization Tips

1. **Use `.shift(1)` for lag operations** - faster than loops
2. **Use `.rolling()` for moving windows** - vectorized
3. **Use `.resample()` for time aggregation** - efficient
4. **Use `.pct_change()` for returns** - no loops needed
5. **Use `.cumprod()` for cumulative returns** - vectorized
6. **Use boolean indexing** - faster than filtering in loops
7. **Use `.loc[]` for label-based indexing** - clearer and faster
8. **Pre-allocate DataFrames** - avoid growing DataFrames in loops

---

**Good luck! You've got this! 🚀**
