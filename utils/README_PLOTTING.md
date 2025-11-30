# Plotting & Formatting Utilities

Standardized plotting and text output utilities to eliminate code duplication across notebooks.

## Quick Start

### Plotting

```python
from utils.plotter import PortfolioPlotter

# Initialize with your backtest results
plotter = PortfolioPlotter(strategy_results, benchmark_data=spy_data)

# Plot equity curves
plotter.plot_equity_curves(log_scale=True)

# Plot drawdown
plotter.plot_drawdown()

# Plot monthly returns heatmap
plotter.plot_monthly_returns_heatmap()

# Plot rolling metrics
plotter.plot_rolling_metrics(window=60, metrics=['sharpe', 'volatility'])

# Comprehensive dashboards
plotter.plot_returns_dashboard(in_sample=True)
plotter.plot_risk_dashboard(in_sample=True)

# Or plot everything at once
plotter.plot_all_dashboards(in_sample=True)
```

### Text Summaries

```python
from utils.formatter import PerformanceSummary

# Initialize with your backtest results
summary = PerformanceSummary(strategy_results, benchmark_data=spy_data, period_label='IN-SAMPLE')

# Print full report
summary.print_full_report()

# Individual sections
summary.print_portfolio_metrics()
summary.print_benchmark_comparison()
summary.print_strategy_rankings()
summary.print_recommendations()

# Trade statistics
summary.print_trade_statistics(strategy_name='Momentum_60')

# Export to DataFrame
df = summary.to_dataframe()
```

### Comparing Periods (IS vs OOS)

```python
from utils.formatter import compare_periods

# Compare in-sample vs out-of-sample
is_summary, oos_summary = compare_periods(
    is_results=strategy_results_is,
    oos_results=strategy_results_oos,
    benchmark_data=spy_data
)
```

## Before vs After

### Before (100+ lines per notebook):

```python
# OLD WAY - Duplicated everywhere
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

for strategy_name, data in strategy_results.items():
    result = data['result']
    equity = result.equity_curve.reset_index()
    axes[0].plot(equity['Date'], 
                 equity['TotalValue'], 
                 label=strategy_name, 
                 linewidth=2)

axes[0].set_title('Individual Strategy Equity Curves', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Equity ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ... 50 more lines for second subplot ...
# ... Another 100+ lines for other plots ...
```

### After (3 lines):

```python
# NEW WAY - Clean and reusable
from utils.plotter import PortfolioPlotter

plotter = PortfolioPlotter(strategy_results)
plotter.plot_equity_curves()
```

## Available Plots

### PortfolioPlotter Methods

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `plot_equity_curves()` | Individual + combined + benchmark equity curves | `log_scale`, `show_benchmark` |
| `plot_drawdown()` | Drawdown % and underwater chart | `show_underwater` |
| `plot_monthly_returns_heatmap()` | Monthly returns as heatmap | `strategy_name` |
| `plot_rolling_metrics()` | Rolling Sharpe, volatility, returns | `window`, `metrics` |
| `plot_returns_dashboard()` | 4-panel returns dashboard | `in_sample` |
| `plot_risk_dashboard()` | 4-panel risk dashboard | `in_sample` |
| `plot_all_dashboards()` | Both returns + risk dashboards | `in_sample` |

### PerformanceSummary Methods

| Method | Description |
|--------|-------------|
| `print_full_report()` | Complete performance report |
| `print_portfolio_metrics()` | Return, Sharpe, drawdown, etc. |
| `print_benchmark_comparison()` | Portfolio vs benchmark |
| `print_strategy_rankings()` | Strategies sorted by performance |
| `print_trade_statistics()` | Win rate, avg win/loss, etc. |
| `print_recommendations()` | Actionable suggestions |
| `print_comparison_table()` | Compare two periods (IS vs OOS) |
| `to_dataframe()` | Export metrics as DataFrame |

## Example: Complete Notebook Cell

```python
from utils.plotter import PortfolioPlotter
from utils.formatter import PerformanceSummary

# Initialize
plotter = PortfolioPlotter(strategy_results, benchmark_data=spy_data)
summary = PerformanceSummary(strategy_results, benchmark_data=spy_data)

# Visualize
plotter.plot_all_dashboards(in_sample=True)

# Summarize
summary.print_full_report()
summary.print_recommendations()
```

That's it! **8 lines instead of 300+**

## Customization

All plotting methods accept standard matplotlib kwargs:

```python
# Custom figure size
plotter.plot_equity_curves(figsize=(20, 12))

# Specific strategies only
plotter.plot_monthly_returns_heatmap(strategy_name='Momentum_120')

# Custom rolling window
plotter.plot_rolling_metrics(window=90, metrics=['sharpe', 'volatility', 'returns'])
```

## Benefits

✅ **Consistency** - Same plots across all notebooks  
✅ **Maintainability** - Fix once, apply everywhere  
✅ **Readability** - Notebooks focus on analysis, not plotting boilerplate  
✅ **Extensibility** - Easy to add new plots to all notebooks  
✅ **Testing** - Plotting logic can be unit tested  

## Integration with Existing Code

These utilities work with your existing `BacktestResult` objects:

```python
# Your existing backtest
pm = PortfolioManagerV2(initial_capital=100000)
result = pm.run_backtest(signals, prices)

# New plotting (works immediately)
strategy_results = {'My_Strategy': {'result': result, 'capital': 100000}}
plotter = PortfolioPlotter(strategy_results)
plotter.plot_equity_curves()
```

## Future Enhancements

- [ ] Plotly versions for interactive HTML reports
- [ ] Additional plots (correlation matrix, trade distribution, etc.)
- [ ] Export plots to PDF/PNG
- [ ] Custom color schemes and themes
- [ ] Signal analysis plots (entry/exit markers, signal strength)
