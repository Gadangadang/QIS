# Modular Portfolio Architecture

## Overview

The portfolio management system has been refactored into a modular architecture with clear separation of concerns. This design follows the **composition over inheritance** pattern, making the system more maintainable, testable, and extensible.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    PortfolioManager                          │
│                  (Top-level Orchestrator)                    │
│                                                              │
│  • Coordinates all components                               │
│  • Manages portfolio state                                  │
│  • Handles signal updates and rebalancing                   │
└──────────┬────────────────┬───────────────┬─────────────────┘
           │                │               │
           ▼                ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │   Risk   │    │ Reporter │    │ Strategy │
    │ Manager  │    │          │    │ Selector │
    └──────────┘    └──────────┘    └──────────┘
```

## Core Components

### 1. PortfolioManager (Implemented)
**File:** `core/portfolio_manager.py`

The top-level orchestrator that manages portfolio state and coordinates all other components.

**Responsibilities:**
- Maintain portfolio state (positions, cash, equity curve)
- Initialize positions based on signals
- Update position values with price changes
- Handle signal changes (entries/exits)
- Execute drift-based rebalancing
- Track trades and transaction costs

**Key Methods:**
```python
pm = PortfolioManager(config)
pm.initialize_positions(prices, signals)
pm.update_positions(prices)
pm.update_signals(signals, prices, date)
pm.rebalance(prices, signals, date)
pm.calculate_metrics()
```

### 2. BacktestResult (Implemented)
**File:** `core/portfolio_manager.py`

Lightweight container for backtest results, useful for walk-forward optimization.

**Responsibilities:**
- Store equity curve and trades
- Calculate performance metrics
- Provide clean interface without full state

**Key Methods:**
```python
result = BacktestResult(equity_df, trades_df, config)
equity = result.get_equity_curve()
trades = result.get_trades_df()
metrics = result.calculate_metrics()
```

### 3. RiskManager (Placeholder)
**File:** `core/risk_manager.py`

Handles position sizing, risk limits, and portfolio constraints.

**Future Responsibilities:**
- Position size calculation (Kelly, fixed fraction, volatility-based)
- Validate trades against risk limits
- Monitor portfolio-wide risk metrics
- Implement stop conditions (max drawdown, etc.)
- Manage correlation-based exposure limits

**Planned Methods:**
```python
risk_mgr = RiskManager(risk_config)
position_size = risk_mgr.calculate_position_size(ticker, signal, capital, positions)
is_valid = risk_mgr.validate_trade(ticker, size, positions, portfolio_value)
should_stop = risk_mgr.check_stop_conditions(equity_curve, initial_capital)
```

### 4. Reporter (Placeholder)
**File:** `core/reporter.py`

Generates reports, visualizations, and analysis output.

**Future Responsibilities:**
- Plot equity curves with drawdown
- Analyze trade distributions
- Format performance metrics
- Generate HTML/PDF reports
- Compare multiple strategies

**Planned Methods:**
```python
reporter = Reporter(output_dir='reports/')
reporter.plot_equity_curve(equity_df, title="Backtest Results")
reporter.plot_trade_analysis(trades_df)
reporter.generate_html_report(equity_df, trades_df, metrics)
reporter.compare_strategies(results_dict)
```

### 5. StrategySelector (Placeholder)
**File:** `core/strategy_selector.py`

Manages walk-forward optimization and multi-strategy selection.

**Future Responsibilities:**
- Walk-forward backtesting framework
- Strategy performance evaluation
- Dynamic strategy selection per asset
- Out-of-sample validation
- Multi-strategy combination

**Planned Methods:**
```python
selector = StrategySelector(wf_config)
results = selector.walk_forward_optimize(strategies, prices, config)
best = selector.select_best_strategy(results, metric='sharpe_ratio')
combined = selector.combine_strategies(strategies, prices, config)
is_valid = selector.validate_out_of_sample(train_result, test_result)
```

## Usage Examples

### Basic Backtest
```python
from core.portfolio_manager import PortfolioManager, PortfolioConfig, run_multi_asset_backtest
from core.multi_asset_loader import load_assets
from signals.momentum import MomentumSignalV2

# Load data
prices = load_assets(['ES', 'GC'], start_date='2015-01-01')

# Generate signals
signal_gen = MomentumSignalV2(lookback=120)
signals = {ticker: signal_gen.generate(prices[ticker]) for ticker in ['ES', 'GC']}

# Configure portfolio
config = PortfolioConfig(
    initial_capital=100000,
    rebalance_threshold=0.05,
    transaction_cost_bps=3.0
)

# Run backtest
result, equity_df, trades_df = run_multi_asset_backtest(signals, prices, config)

# Get metrics
metrics = result.calculate_metrics()
print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
print(f"CAGR: {metrics['CAGR']:.2%}")
```

### With Risk Management (Future)
```python
from core.risk_manager import RiskManager, RiskConfig

# Configure risk management
risk_config = RiskConfig(
    max_position_size=0.20,
    position_sizing_method='kelly',
    kelly_fraction=0.25,
    max_drawdown_stop=0.20
)

risk_mgr = RiskManager(risk_config)

# Risk manager will be integrated into PortfolioManager
# to validate and size positions
```

### With Reporting (Future)
```python
from core.reporter import Reporter

reporter = Reporter(output_dir='reports/')

# Generate comprehensive report
reporter.generate_html_report(
    equity_df=equity_df,
    trades_df=trades_df,
    metrics=metrics,
    title="ES+GC Momentum Strategy",
    save_path="reports/momentum_strategy.html"
)

# Compare strategies
reporter.compare_strategies({
    'Momentum': momentum_equity_df,
    'Mean Reversion': mr_equity_df
})
```

### Walk-Forward Optimization (Future)
```python
from core.strategy_selector import StrategySelector, WalkForwardConfig
from signals.momentum import MomentumSignalV2
from signals.mean_reversion import MeanReversionSignal

# Define strategies to test
strategies = {
    'Momentum_120': MomentumSignalV2(lookback=120),
    'Momentum_60': MomentumSignalV2(lookback=60),
    'MeanRev': MeanReversionSignal(lookback=20, entry_z=2.0)
}

# Configure walk-forward
wf_config = WalkForwardConfig(
    train_period_days=252,  # 1 year train
    test_period_days=63,    # 1 quarter test
    selection_metric='sharpe_ratio',
    min_trades=10
)

selector = StrategySelector(wf_config)

# Run walk-forward optimization
results = selector.walk_forward_optimize(
    strategies=strategies,
    prices_dict=prices,
    portfolio_config=config
)

# Select best strategy per period
for period, result in results.items():
    best = selector.select_best_strategy(result)
    print(f"{period}: Best strategy = {best}")
```

## Design Principles

### 1. Separation of Concerns
Each component has a single, well-defined responsibility:
- **PortfolioManager**: Portfolio state and execution
- **RiskManager**: Risk rules and limits
- **Reporter**: Output and visualization
- **StrategySelector**: Strategy evaluation and selection

### 2. Composition Over Inheritance
Components are composed together rather than using deep inheritance hierarchies:
```python
class PortfolioManager:
    def __init__(self, config, risk_manager=None, reporter=None):
        self.config = config
        self.risk_mgr = risk_manager  # Optional composition
        self.reporter = reporter      # Optional composition
```

### 3. Clean Interfaces
Each component has clear inputs and outputs:
- Configuration via dataclasses
- Return typed results (BacktestResult, not tuples)
- No hidden state or side effects

### 4. Testability
Each component can be tested in isolation:
- Mock dependencies as needed
- Clear success/failure conditions
- Deterministic behavior

### 5. Extensibility
Easy to add new components or features:
- Add new strategies without modifying core
- Plug in different risk managers
- Extend reporting formats
- Add new optimization methods

## Migration Path

The refactoring preserves backward compatibility while enabling gradual migration:

### Current State (Implemented)
✅ Clean modular structure with PortfolioManager and BacktestResult
✅ All existing tests pass with identical results
✅ Fast performance (0.26s for 10-year backtest)

### Next Steps
1. **Implement Reporter** - Visualization and HTML reports
2. **Implement RiskManager** - Position sizing and risk limits
3. **Implement StrategySelector** - Walk-forward optimization
4. **Integrate Components** - Wire components into PortfolioManager
5. **Real-time Support** - Extend for live trading

## Benefits

### For Development
- **Maintainability**: Changes to one component don't affect others
- **Testability**: Each component can be unit tested
- **Clarity**: Clear responsibilities and interfaces
- **Reusability**: Components can be used in different contexts

### For Research
- **Flexibility**: Easy to swap components (e.g., different risk managers)
- **Experimentation**: Test new ideas without breaking existing code
- **Analysis**: Reporter provides consistent output format
- **Optimization**: StrategySelector enables systematic testing

### For Production
- **Reliability**: Well-tested, modular components
- **Monitoring**: Clear separation of concerns for logging/monitoring
- **Scalability**: Easy to parallelize walk-forward optimization
- **Maintenance**: Clear ownership and boundaries

## File Organization

```
core/
├── portfolio_manager.py      # ✅ Main orchestrator + BacktestResult
├── risk_manager.py            # ⏳ Position sizing and risk limits
├── reporter.py                # ⏳ Visualization and reports
└── strategy_selector.py       # ⏳ Walk-forward optimization

signals/
├── base.py                    # Signal interface
├── momentum.py                # Momentum strategies
└── mean_reversion.py          # Mean reversion strategies

utils/
├── metrics.py                 # Performance calculations
└── logger.py                  # Logging utilities
```

Legend:
- ✅ Implemented and tested
- ⏳ Placeholder with design (ready for implementation)
- 📝 Planned

## Conclusion

The new modular architecture provides a solid foundation for:
- Multi-strategy portfolio management
- Walk-forward optimization
- Risk management
- Professional reporting
- Live trading (future)

The system is production-ready for backtesting while providing clear extension points for advanced features.
