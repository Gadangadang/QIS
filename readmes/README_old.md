# QuantTrading

A multi-asset, multi-strategy quantitative portfolio management system with walk-forward optimization, comprehensive risk controls, and production-ready backtesting infrastructure.

## 🎯 Overview

Professional-grade quantitative trading framework demonstrating:
- **Multi-asset portfolio management** (equity indices, commodities, futures)
- **Multi-strategy framework** (different strategies per asset for true diversification)
- **Walk-forward optimization** with per-fold diagnostics
- **Advanced portfolio construction** (risk budgeting, dynamic position sizing, rebalancing)
- **Comprehensive analytics** (performance attribution, regime analysis, correlation studies)

**Current Status:** Research and backtesting infrastructure complete. Paper trading capabilities available (live trading restricted by compliance).

## 📊 Quick Start

### 1. Multi-Asset Portfolio Backtest

```python
from core.portfolio_manager import PortfolioManager
from core.multi_asset_loader import load_futures_data
from core.multi_strategy_signal import StrategyConfig

# Load multiple assets
prices = load_futures_data(['ES', 'NQ', 'GC'], start_date='2000-01-01')

# Configure different strategies per asset
config = (StrategyConfig()
          .add_momentum('ES', lookback=120, entry_threshold=0.02)
          .add_momentum('NQ', lookback=90, entry_threshold=0.03)
          .add_mean_reversion('GC', window=50, entry_z=2.0, exit_z=0.5)
          .build())

# Generate signals
signals = config.generate(prices)

# Run portfolio backtest with risk controls
portfolio_config = {
    'initial_capital': 100000,
    'risk_per_trade': 0.02,
    'max_position_size': 0.20,
    'rebalance_frequency': 'monthly',
    'transaction_cost_bps': 3.0
}

pm = PortfolioManager(**portfolio_config)
equity_curve, trades = pm.run_backtest(signals, prices)

# Analyze results
print(f"Final Equity: ${equity_curve.iloc[-1]:,.2f}")
print(f"Total Trades: {len(trades)}")
print(f"Sharpe Ratio: {calculate_sharpe(equity_curve):.2f}")
```

### 2. Walk-Forward Optimization

```python
from core.optimizer import WalkForwardOptimizer

# Define parameter grid
param_grid = {
    'lookback': [60, 90, 120, 150],
    'entry_threshold': [0.01, 0.02, 0.03],
    'exit_threshold': [-0.01, 0.0, 0.01]
}

# Run optimization with 5 folds
optimizer = WalkForwardOptimizer(
    signal_class='momentum',
    param_grid=param_grid,
    n_folds=5,
    metric='sharpe'
)

results = optimizer.optimize(prices)
best_params = optimizer.get_best_params()
```

### 3. Explore Research Notebooks

```bash
jupyter lab notebooks/05_multi_asset_demo.ipynb
```

Comprehensive notebooks with:
- Multi-asset portfolio construction
- Signal correlation analysis
- Walk-forward validation results
- Performance attribution by asset
- Rebalancing visualization

## 🏗️ Project Structure

```
QuantTrading/
├── core/                           # Core trading engine
│   ├── portfolio_manager.py       # Multi-asset portfolio management
│   ├── multi_strategy_signal.py   # Different strategies per asset
│   ├── multi_asset_loader.py      # Futures data loader (ES, NQ, GC, etc.)
│   ├── optimizer.py               # Walk-forward optimization
│   ├── position_sizers.py         # Risk-based position sizing
│   ├── backtest_engine.py         # Single-asset backtesting (legacy)
│   └── paper_trader.py            # Paper trading simulator
│
├── signals/                        # Trading strategies
│   ├── base.py                    # SignalModel abstract base class
│   ├── momentum.py                # Trend-following strategies
│   ├── mean_reversion.py          # Counter-trend strategies
│   └── ensemble.py                # Multi-timeframe ensembles
│
├── analysis/                       # Analytics and reporting
│   ├── metrics.py                 # Performance metrics (Sharpe, Sortino, etc.)
│   └── report.py                  # HTML report generation
│
├── utils/                          # Utilities
│   └── logger.py                  # Logging configuration
│
├── notebooks/                      # Research notebooks
│   ├── 05_multi_asset_demo.ipynb  # Multi-asset portfolio (Week 4-5)
│   ├── 04_position_sizing_optimization.ipynb
│   └── 03_backtest_momentum.ipynb # Single-asset research
│
├── scripts/                        # Test and utility scripts
│   ├── test_multi_strategy.py     # Multi-strategy validation
│   ├── test_portfolio_allocation.py
│   ├── test_perfold_optimization.py
│   ├── test_optimizer.py          # Walk-forward optimizer tests
│   └── run_daily.py               # Daily execution script
│
├── Dataset/                        # Market data
│   ├── spx_data.csv              # S&P 500 futures (ES)
│   ├── nq_data.csv               # NASDAQ futures (NQ)
│   └── gc_data.csv               # Gold futures (GC)
│
├── logs/                           # Backtest results
│   └── [experiment_name]/
│       ├── equity_curve.csv
│       ├── trades.csv
│       ├── diagnostics.txt        # Per-fold optimization details
│       └── report.html
│
├── archive/                        # Deprecated code
│   └── old_structure/             # Pre-refactor code (can be deleted)
│
└── Documentation
    ├── README.md                  # This file
    ├── IMMEDIATE_TODOS.md         # Development roadmap
    ├── OPTIMIZER_USAGE.md         # Optimizer guide
    └── STRATEGIC_PLAN.md          # Long-term vision
```

## 🚀 Key Features

### 1. Multi-Strategy Framework

Apply different strategies to different assets for true diversification:

```python
# Momentum for equities, mean reversion for commodities
config = (StrategyConfig()
          .add_momentum('ES', lookback=120)      # S&P 500: trend-following
          .add_momentum('NQ', lookback=90)       # NASDAQ: trend-following
          .add_mean_reversion('GC', window=50)   # Gold: counter-trend
          .build())
```

**Why this matters:**
- Signal correlation > Return correlation for diversification
- Using same strategy across assets = highly correlated signals (95%+)
- Different strategies = lower signal correlation = better diversification

### 2. Portfolio Management

Professional risk management and position sizing:

- **Risk-based position sizing:** Fixed risk per trade (e.g., 2% of capital)
- **Dynamic rebalancing:** Monthly/quarterly/custom frequencies
- **Portfolio constraints:** Max position size per asset, sector limits
- **Transaction costs:** Realistic cost modeling (slippage + commissions)
- **Correlation monitoring:** Track signal and return correlations

### 3. Walk-Forward Optimization

Robust parameter selection without lookahead bias:

- **Anchored walk-forward:** Expanding training window
- **Per-fold diagnostics:** Track performance by fold
- **Multiple metrics:** Optimize on Sharpe, Sortino, Calmar, or custom
- **Strategy selection:** Automatically choose best strategy per asset per period
- **Performance attribution:** Understand what works when

### 4. Comprehensive Analytics

Deep insights into strategy performance:

- **Performance metrics:** Sharpe, Sortino, Calmar, CAGR, Max Drawdown
- **Trade analysis:** Win rate, profit factor, avg win/loss, duration
- **Regime analysis:** Performance in different market conditions
- **Correlation studies:** Signal correlation, return correlation, factor exposure
- **Attribution:** Performance by asset, by strategy, by time period

## 📚 Usage Examples

### Multi-Asset Portfolio Construction

```python
from core.portfolio_manager import PortfolioManager
from core.multi_asset_loader import load_futures_data
from core.multi_strategy_signal import StrategyConfig

# Load data for multiple assets
prices = load_futures_data(
    tickers=['ES', 'NQ', 'GC'],
    start_date='2000-01-01',
    end_date='2024-12-31'
)

# Configure strategy per asset
config = (StrategyConfig()
    .add_momentum('ES', lookback=120, entry_threshold=0.02, exit_threshold=-0.01)
    .add_momentum('NQ', lookback=90, entry_threshold=0.03, exit_threshold=0.0)
    .add_mean_reversion('GC', window=50, entry_z=2.0, exit_z=0.5)
    .build())

# Generate signals
signals = config.generate(prices)

# Configure portfolio
portfolio_config = {
    'initial_capital': 100000,
    'risk_per_trade': 0.02,              # 2% risk per trade
    'max_position_size': 0.20,           # Max 20% in any single asset
    'rebalance_frequency': 'monthly',
    'transaction_cost_bps': 3.0
}

# Run backtest
pm = PortfolioManager(**portfolio_config)
equity_curve, trades = pm.run_backtest(signals, prices)

# Analyze
from analysis.metrics import calculate_sharpe, calculate_max_drawdown
print(f"Sharpe: {calculate_sharpe(equity_curve):.2f}")
print(f"Max DD: {calculate_max_drawdown(equity_curve):.1%}")
print(f"Trades: {len(trades)}")
```

### Walk-Forward Optimization

```python
from core.optimizer import WalkForwardOptimizer

# Define parameter space
param_grid = {
    'lookback': [60, 90, 120, 150],
    'entry_threshold': [0.01, 0.02, 0.03, 0.05],
    'exit_threshold': [-0.02, -0.01, 0.0, 0.01]
}

# Initialize optimizer
optimizer = WalkForwardOptimizer(
    signal_class='momentum',
    param_grid=param_grid,
    n_folds=5,
    train_fraction=0.6,
    test_fraction=0.2,
    metric='sharpe',
    save_dir='logs/wf_optimization'
)

# Run optimization
results = optimizer.optimize(prices['ES'])

# Get best parameters for each fold
best_params = optimizer.get_best_params()
print(best_params)

# Analyze per-fold performance
diagnostics = optimizer.load_diagnostics()
```

### Strategy Research in Notebooks

See `notebooks/05_multi_asset_demo.ipynb` for comprehensive examples including:
- Signal correlation analysis
- Performance attribution by asset
- Rebalancing visualization
- Regime-dependent performance
- Walk-forward validation results

## 📖 Documentation

- **[Core Multi-Strategy Framework](core/README_MULTI_STRATEGY.md)** - Detailed guide to multi-strategy system
- **[Optimizer Usage Guide](OPTIMIZER_USAGE.md)** - Walk-forward optimization examples
- **[Development Roadmap](IMMEDIATE_TODOS.md)** - Current priorities and future plans
- **[Strategic Plan](STRATEGIC_PLAN.md)** - Long-term vision and architecture

## 🧪 Testing

### Quick Test
```bash
python scripts/test_multi_strategy.py
```

Validates:
- Multi-asset data loading
- Strategy configuration
- Portfolio management
- Signal generation
- Trade execution

### Comprehensive Tests
```bash
# Test portfolio allocation
python scripts/test_portfolio_allocation.py

# Test walk-forward optimizer
python scripts/test_optimizer.py

# Test position sizing strategies
python scripts/test_position_sizing.py
```

## 🔧 Installation

### Requirements
- Python 3.9+
- pandas, numpy
- matplotlib, seaborn
- jupyter/jupyterlab (for notebooks)
- optuna (for optimization)

### Setup
```bash
# Clone repository
git clone https://github.com/Gadangadang/QuantTrading.git
cd QuantTrading

# Create environment using conda (recommended)
conda env create -f environment.yml
conda activate quant_trading

# Or use pip
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn jupyterlab optuna
```

## 📊 Performance Summary

Multi-asset portfolio results (ES + NQ + GC, 2000-2024):

**All-Momentum Strategy:**
- 6 rebalances over 25 years
- Signal correlation: 0.95+ (highly correlated)
- Limited diversification benefit despite negative GC-equity return correlation

**Multi-Strategy Framework:**
- 311 trades over 10 years (50x more activity)
- Signal correlation: Significantly reduced
- GC behavior: 89% long (momentum) → 16.3% long (mean reversion)
- More diversified signal exposure

**Key Insight:** Signal correlation matters more than return correlation for diversification.

## 🔜 Next Steps

1. **Performance Optimization** ⚡
   - Vectorize portfolio calculations
   - Profile and optimize bottlenecks
   - Target: <5 seconds for 10-year, 3-asset backtest

2. **Walk-Forward Multi-Strategy** 🎯
   - Optimize strategy selection per asset per period
   - Visualize strategy allocation over time
   - Compare adaptive vs. static allocation

3. **Advanced Features**
   - Regime detection and regime-dependent strategies
   - Factor-based position sizing
   - Portfolio constraints and risk budgeting
   - Real-time monitoring dashboard

## 🤝 Contributing

This is a personal research project demonstrating quantitative portfolio management skills. Not currently accepting external contributions.

## 📝 License

Private research project. All rights reserved.



