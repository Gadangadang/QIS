# QuantTrading

[![Tests](https://github.com/Gadangadang/QuantTrading/actions/workflows/test.yml/badge.svg)](https://github.com/Gadangadang/QuantTrading/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/Gadangadang/QuantTrading/branch/main/graph/badge.svg)](https://codecov.io/gh/Gadangadang/QuantTrading)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: Private](https://img.shields.io/badge/license-Private-red.svg)]()

A professional-grade multi-asset, multi-strategy quantitative portfolio management system with advanced signal generation, ensemble methods, comprehensive risk controls, and production-ready backtesting infrastructure.

## 🎯 Overview

Institutional-quality quantitative trading framework featuring:
- **Multi-asset portfolio management** (equity indices, commodities, futures)
- **Advanced signal strategies** (trend-following long-short, adaptive ensembles, regime-aware signals)
- **Modular portfolio system** (v2 architecture with clean separation of concerns)
- **Comprehensive risk management** (dynamic position sizing, stop-loss, drawdown controls)
- **Professional reporting** (interactive HTML dashboards with Plotly visualizations)
- **Benchmark analysis** (SPY comparison with alpha/beta metrics, rolling correlations)

**Current Status:** Production-ready backtesting infrastructure with paper trading capabilities. Live trading restricted by compliance.

## 🚀 Key Features

### 1. Advanced Signal Strategies

**Trend Following Long-Short:**
- Multi-timeframe momentum confirmation (fast + slow)
- Volume and volatility regime filters
- Can go LONG, SHORT, or FLAT (cash during uncertainty)
- Designed to capture bear markets and avoid choppy periods

**Adaptive Ensemble:**
- Dynamically weights multiple strategies based on rolling Sharpe ratios
- Adapts to changing market regimes (bull/bear/sideways)
- Combines momentum, trend-following, mean-reversion
- Performance-based allocation with signal strength filtering

**Momentum V2:**
- Enhanced momentum with entry/exit thresholds
- Configurable lookback periods
- Risk-adjusted position sizing

### 2. Refactored Portfolio System (V2)

Clean, modular architecture with separation of concerns:

```
PortfolioManagerV2
├── Portfolio (state management)
│   ├── Positions tracking
│   ├── Cash management
│   └── Portfolio value calculation
│
├── RiskManager (risk controls)
│   ├── Position sizing (% risk per trade)
│   ├── Max position limits
│   ├── Stop-loss enforcement
│   └── Drawdown monitoring
│
├── ExecutionEngine (order execution)
│   ├── Transaction costs (3 bps)
│   ├── Slippage modeling (2 bps)
│   └── Realistic fills
│
└── BacktestResult (analysis)
    ├── Performance metrics (Sharpe, Sortino, Calmar)
    ├── Trade analysis
    └── HTML report generation
```

### 3. Multi-Strategy Framework

Apply different strategies to different assets for true diversification:

```python
strategies = [
    {
        'name': 'Adaptive_Ensemble',
        'signal_generator': AdaptiveEnsemble([
            ('momentum', MomentumSignalV2(lookback=60), 0.33),
            ('trend_ls', TrendFollowingLongShort(), 0.34),
            ('adaptive_trend', AdaptiveTrendFollowing(), 0.33)
        ]),
        'assets': ['ES', 'GC'],
        'capital': 50000
    },
    {
        'name': 'TrendFollowing_LS',
        'signal_generator': TrendFollowingLongShort(),
        'assets': ['NQ'],
        'capital': 30000
    }
]
```

**Why this matters:**
- Signal correlation > Return correlation for diversification
- Different strategies per asset = lower signal correlation = better diversification
- 311 trades vs 6 rebalances (50x more signal diversity)

### 4. Professional Reporting System

**Performance Report:**
- Interactive equity curves (portfolio + strategies + SPY benchmark)
- Rolling beta analysis (90-day window)
- Normalized returns comparison (base 100)
- Trade P&L distribution
- Comprehensive metrics tables

**Risk Dashboard:**
- Underwater drawdown chart with equity peaks
- Strategy correlation matrix (4x4 heatmap)
- Rolling risk metrics (30/60/90-day volatility, Sharpe ratio)
- Value at Risk (VaR) and Conditional VaR analysis
- Returns distribution histogram

All charts use `.tolist()` serialization to avoid Plotly pandas bugs.

### 5. Benchmark Analysis

Compare against SPY with institutional metrics:
- **Alpha & Beta**: Risk-adjusted excess returns
- **Information Ratio**: Consistency of outperformance
- **Up/Down Capture**: Performance in bull/bear markets
- **Rolling Correlations**: Dynamic relationship tracking
- **Base 100 Normalization**: Visual comparison of growth

## 📊 Quick Start

### 1. Simple Backtest with Futures Contract Sizing

```python
from core.portfolio.portfolio_manager_v2 import PortfolioManagerV2
from core.portfolio.position_sizers import FuturesContractSizer
from core.multi_asset_loader import load_assets
from signals.momentum import MomentumSignalV2

# Define contract multipliers for futures
CONTRACT_MULTIPLIERS = {
    'CL': 1000,   # Crude Oil (1,000 barrels per contract)
    'NG': 10000,  # Natural Gas (10,000 MMBtu per contract)
    'GC': 100,    # Gold (100 troy ounces per contract)
    'ES': 50,     # S&P 500 E-mini (50x index value)
    'NQ': 20      # Nasdaq 100 E-mini (20x index value)
}

# Load data
prices = load_assets(['CL', 'NG', 'GC'], start_date='2017-01-01', use_yfinance=True)

# Create futures contract sizer (ensures integer contracts)
futures_sizer = FuturesContractSizer(
    contract_multipliers=CONTRACT_MULTIPLIERS,
    max_position_pct=0.25,
    risk_per_trade=0.02,
    min_contracts=1
)

# Generate signals
signal_gen = MomentumSignalV2(lookback=60, entry_threshold=0.02)
signals = {ticker: signal_gen.generate(prices[ticker]) for ticker in prices.keys()}

# Run backtest with integer contract sizing
pm = PortfolioManagerV2(
    initial_capital=500_000,
    risk_per_trade=0.02,
    max_position_size=0.25,
    transaction_cost_bps=3.0,
    slippage_bps=2.0,
    position_sizer=futures_sizer  # Use futures contract sizer
)

result = pm.run_backtest(signals, prices)
result.print_summary()
```

### 2. Multi-Strategy Portfolio with Clean Architecture

```python
from utils.plotter import PortfolioPlotter
from utils.formatter import PerformanceSummary

# Configure strategies
strategies = [
    {
        'name': 'Commodities_HybridAdaptive',
        'signal_generator': HybridAdaptiveSignal(
            vol_window=30, mr_window=15, mom_fast=30, mom_slow=90
        ),
        'assets': ['CL', 'NG', 'GC'],
        'capital': 500_000
    },
    {
        'name': 'Equities_Momentum',
        'signal_generator': MomentumSignalV2(
            lookback=60, entry_threshold=0.02
        ),
        'assets': ['ES', 'NQ'],
        'capital': 500_000
    }
]

# Run all strategies
strategy_results = {}
for strat in strategies:
    signals = {asset: strat['signal_generator'].generate(prices[asset]) 
               for asset in strat['assets']}
    
    pm = PortfolioManagerV2(
        initial_capital=strat['capital'],
        position_sizer=futures_sizer
    )
    result = pm.run_backtest(signals, prices)
    strategy_results[strat['name']] = {
        'result': result,
        'capital': strat['capital'],
        'assets': strat['assets']
    }

# Generate visualizations using utilities
plotter = PortfolioPlotter(strategy_results)
plotter.plot_equity_curves(show_individual=True, show_combined=True)

# Load benchmark and compare
benchmark_loader = BenchmarkLoader(cache_dir="Dataset")
benchmark_data = benchmark_loader.load_benchmark('SPY', start_date='2017-01-01')

summary = PerformanceSummary(
    strategy_results, 
    benchmark_data=benchmark_data, 
    period_label='IN-SAMPLE'
)
summary.print_benchmark_comparison()
summary.print_strategy_rankings()
```

### 3. Explore Research Notebooks

```bash
jupyter lab notebooks/multi_strategy_refactored.ipynb
```

Comprehensive notebooks demonstrating:
- **multi_strategy_refactored.ipynb** - Clean multi-strategy architecture with ensemble methods
- **multi_strategy_commodities.ipynb** - Futures trading with integer contract sizing
- **oil_gas_exploration.ipynb** - Oil & gas futures development notebook

Key features showcased:
- Multi-strategy portfolio construction
- Futures contract position sizing
- Adaptive ensemble configuration
- Benchmark comparison (SPY)
- Performance attribution using PortfolioPlotter
- Risk analysis using PerformanceSummary
- Signal correlation studies

## 🏗️ Project Structure

```
QuantTrading/
├── core/                                  # Core trading engine
│   ├── portfolio/                         # V2 Portfolio System (modular)
│   │   ├── portfolio_manager_v2.py       # Main orchestrator
│   │   ├── portfolio.py                  # State management
│   │   ├── risk_manager.py               # Risk controls
│   │   ├── execution_engine.py           # Order execution
│   │   ├── backtest_result.py            # Results container
│   │   └── position_sizers.py            # Position sizing strategies
│   │
│   ├── futures/                           # Futures infrastructure
│   │   ├── rollover_handler.py           # Contract rollover logic
│   │   └── __init__.py
│   │
│   ├── asset_registry.py                 # Asset metadata registry
│   ├── multi_asset_loader.py             # Multi-asset data loader
│   ├── multi_strategy_reporter.py        # Performance reports (HTML)
│   ├── risk_dashboard.py                 # Risk analysis dashboard
│   ├── benchmark.py                      # Benchmark comparison tools
│   └── paper_trading_engine.py           # Live paper trading
│
├── signals/                               # Trading strategies
│   ├── base.py                           # SignalModel abstract base
│   ├── momentum.py                       # Momentum strategies
│   ├── mean_reversion.py                 # Counter-trend strategies
│   ├── hybrid_adaptive.py                # Hybrid adaptive signals
│   ├── trend_following_long_short.py     # Long-short trend
│   └── ensemble.py                       # Adaptive ensembles
│
├── utils/                                 # Utilities
│   ├── plotter.py                        # Visualization utilities
│   ├── formatter.py                      # Performance formatters
│   └── logger.py                         # Logging
│
├── notebooks/                             # Research notebooks
│   ├── multi_strategy_refactored.ipynb   # Main multi-strategy demo
│   ├── multi_strategy_commodities.ipynb  # Commodities futures
│   └── oil_gas_exploration.ipynb         # Oil & gas development
│
├── tests/                                 # Test suite
│   ├── test_portfolio_core.py            # Portfolio & risk tests
│   ├── conftest.py                       # Test fixtures
│   └── README.md                         # Test documentation
│
├── readmes/                               # Documentation
│   ├── COMMODITIES_EXPANSION_PLAN.md     # Commodities roadmap
│   └── OIL_GAS_IMPLEMENTATION.md         # Oil/gas implementation
│
├── Dataset/                               # Market data
│   ├── spx_data_v1.csv                   # S&P 500 futures (ES)
│   ├── fix_data.py                       # Data cleaning scripts
│   └── energy/                           # Commodity data (planned)
│
├── backtest/                              # Legacy backtest engine
├── live/                                  # Paper trading scripts
├── logs/                                  # Trading logs
└── config/                                # Configuration files
```

## 💡 Usage Examples

### Example 1: Test Trend Following Long-Short

```python
from core.portfolio.portfolio_manager_v2 import PortfolioManagerV2
from signals.trend_following_long_short import TrendFollowingLongShort
from core.benchmark import BenchmarkLoader, BenchmarkComparator

# Load SPY data
prices = load_assets(['SPY'], start_date='2015-01-01')

# Generate signals
signal = TrendFollowingLongShort(
    fast_period=20,
    slow_period=100,
    momentum_threshold=0.02,
    volume_multiplier=1.1,
    vol_percentile=0.70
)

signals = {'SPY': signal.generate(prices['SPY'])}

# Backtest
pm = PortfolioManagerV2(initial_capital=100000)
result = pm.run_backtest(signals, prices)

# Compare to benchmark
benchmark = BenchmarkLoader().load_benchmark('SPY')
comparator = BenchmarkComparator()
metrics = comparator.calculate_metrics(
    result.equity_curve,
    benchmark,
    risk_free_rate=0.02
)

print(f"Portfolio Return: {metrics['Portfolio Return']:.2%}")
print(f"Benchmark Return: {metrics['Benchmark Return']:.2%}")
print(f"Alpha: {metrics['Alpha (Annual)']:.2%}")
print(f"Beta: {metrics['Beta (Full Period)']:.3f}")
print(f"Information Ratio: {metrics['Information Ratio']:.2f}")
```

### Example 2: Adaptive Ensemble on Multiple Assets

```python
from signals.ensemble import AdaptiveEnsemble
from signals.momentum import MomentumSignalV2
from signals.trend_following_long_short import TrendFollowingLongShort

# Create adaptive ensemble
ensemble = AdaptiveEnsemble(
    strategies=[
        ('momentum', MomentumSignalV2(lookback=60, entry_threshold=0.02), 0.33),
        ('trend_ls', TrendFollowingLongShort(fast_period=20, slow_period=100), 0.34),
        ('adaptive_trend', AdaptiveTrendFollowing(base_period=60), 0.33)
    ],
    method='adaptive',              # Use performance-based weighting
    adaptive_lookback=60,           # 60-day performance window
    signal_threshold=0.3,           # 30% minimum confidence
    rebalance_frequency=20          # Update weights every 20 days
)

# Apply to multiple assets
prices = load_assets(['ES', 'GC', 'NQ'])
signals = {
    asset: ensemble.generate(prices[asset])
    for asset in prices.keys()
}

# Run backtest
result = pm.run_backtest(signals, prices)
```

### Example 3: Multi-Strategy Portfolio with HTML Reports

```python
# Configure multiple strategies
strategies = [
    {
        'name': 'Adaptive_Ensemble',
        'signal_generator': AdaptiveEnsemble([...]),
        'assets': ['ES', 'GC'],
        'capital': 50000
    },
    {
        'name': 'TrendFollowing_LS',
        'signal_generator': TrendFollowingLongShort(),
        'assets': ['NQ'],
        'capital': 30000
    },
    {
        'name': 'Classic_Momentum',
        'signal_generator': MomentumSignalV2(lookback=60),
        'assets': ['ES'],
        'capital': 20000
    }
]

# Run all strategies
strategy_results = {}
for strat in strategies:
    signal_dict = {asset: strat['signal_generator'].generate(prices[asset]) 
                   for asset in strat['assets']}
    pm = PortfolioManagerV2(initial_capital=strat['capital'])
    result = pm.run_backtest(signal_dict, prices)
    strategy_results[strat['name']] = {'result': result, 'capital': strat['capital']}

# Generate reports
from core.multi_strategy_reporter import MultiStrategyReporter
from core.risk_dashboard import RiskDashboard

reporter = MultiStrategyReporter()
risk_dash = RiskDashboard()

# Load benchmark
benchmark = BenchmarkLoader().load_benchmark('SPY')

# Generate HTML
perf_html = reporter.generate_report(
    strategy_results=strategy_results,
    combined_equity=combined_equity,
    benchmark_data=benchmark,
    benchmark_name='SPY'
)

risk_html = risk_dash.generate_dashboard(
    strategy_results=strategy_results,
    combined_equity=combined_equity,
    benchmark_data=benchmark
)

# Save reports
with open('reports/performance.html', 'w') as f:
    f.write(perf_html)
with open('reports/risk.html', 'w') as f:
    f.write(risk_html)
```

## 📈 Performance Summary

**Multi-Strategy Portfolio (2015-2024):**
- **Combined Return:** 147.80%
- **SPY Benchmark:** 240.81%
- **Sharpe Ratio:** ~1.8 (portfolio-weighted average)
- **Max Drawdown:** Better risk management than buy-and-hold
- **Beta:** 0.364 (lower volatility than market)
- **Trades:** 311 trades vs 6 rebalances (better signal diversification)

**Key Insights:**
- Signal correlation matters more than return correlation
- Adaptive ensemble adjusts to market regimes
- Long-short capability reduces drawdowns
- Multiple strategies provide true diversification

## 🔧 Installation

### Requirements
- Python 3.9+
- See `requirements.txt` for full dependency list

### Quick Setup (pip)
```bash
# Clone repository
git clone https://github.com/Gadangadang/QuantTrading.git
cd QuantTrading

# Install production dependencies
pip install -r requirements.txt

# Or install with development tools
pip install -r requirements-dev.txt
```

### Conda Setup (Recommended)
```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate quant_trading

# Or create fresh environment
conda create -n quant_trading python=3.11
conda activate quant_trading
pip install -r requirements.txt
```

### Verify Installation
```bash
# Run tests to verify everything works
pytest tests/ -v

# Or check imports
python -c "import pandas, numpy, matplotlib, plotly; print('✅ All dependencies installed')"
```

## 🧪 Testing

### Unit Tests

**Current Coverage: 51%** (376/733 lines tested)

Run the test suite:
```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=core/portfolio --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=core/portfolio --cov-report=html
```

**Test Suite:**
- ✅ 21 passing tests
- Position sizing strategies (FixedFractional, Kelly, ATR)
- Risk management (stop-loss, take-profit, concentration limits)
- Portfolio tracking (positions, P&L, cash management)
- Integration tests (full backtest workflow)

**Module Coverage:**
- `execution_engine.py`: 84% ✅
- `portfolio.py`: 76% ✅
- `position_sizers.py`: 65% ⚠️
- `portfolio_manager_v2.py`: 47% ⚠️
- `risk_manager.py`: 35% (more tests coming)

See `tests/README.md` for detailed documentation.

### Continuous Integration

GitHub Actions automatically runs tests on every push:
- ✅ Tests across Python 3.9, 3.10, 3.11
- ✅ Coverage reporting with Codecov
- ✅ Code quality checks (black, flake8, isort)
- ✅ 50% minimum coverage requirement

**CI Pipeline:** `.github/workflows/test.yml`

**Dependabot:** Automated weekly dependency updates

### Quick Test
```bash
# Test multi-strategy system
python -m pytest tests/ -v

# Or run manual validation
python scripts/test_multi_strategy.py
```

### Comprehensive Tests
```python
# Test in notebook
jupyter lab notebooks/multi_strategy_with_ensemble.ipynb
```

Validates:
- Multi-asset data loading
- Signal generation (all types)
- Portfolio management V2
- Risk controls
- Trade execution
- Report generation
- Benchmark comparison

## 📖 Documentation

- **[New Signals Guide](signals/README_NEW_SIGNALS.md)** - Trend-following long-short & adaptive ensembles
- **[Portfolio V2 Architecture](core/portfolio/)** - Modular system design
- **[Signal Development](signals/README.md)** - Creating new strategies
- **[Backtest Engine](core/BACKTEST_README.md)** - Engine documentation

## 🔜 Next Steps

### Immediate
- [x] Fix Plotly serialization bugs (`.tolist()` for all charts)
- [x] Adaptive ensemble with dynamic weighting
- [x] Trend-following long-short signals
- [x] Comprehensive HTML reporting with risk dashboard
- [x] Benchmark comparison (SPY alpha/beta analysis)

### In Progress
- [ ] Walk-forward optimization for multi-strategy portfolios
- [ ] Parameter tuning (grid search with cross-validation)
- [ ] Regime detection (bull/bear/sideways classification)
- [ ] Factor-based position sizing

### Future
- [ ] Real-time data integration
- [ ] Live paper trading monitoring dashboard
- [ ] Machine learning signal ensembles
- [ ] Options strategies integration

## 🚨 Important Notes

### Data Requirements
- **Minimum:** 2 years daily data (for adaptive signals)
- **Recommended:** 5+ years for proper validation
- **Optimal:** 10+ years to test through multiple market cycles

### Transaction Costs
All backtests include realistic costs:
- Transaction costs: 3 bps per trade
- Slippage: 2 bps per trade
- Total: ~5 bps round-trip (conservative for liquid futures)

### Chart Rendering
Reports use `.tolist()` to serialize pandas data for Plotly, avoiding serialization bugs. Hard refresh browser (`Cmd+Shift+R`) if charts don't update.

## 🤝 Contributing

This is a personal research project demonstrating quantitative portfolio management and signal development skills. Not currently accepting external contributions.

## 📝 License

Private research project. All rights reserved.

---

**Built with:** Python, Pandas, NumPy, Plotly, Matplotlib, SciPy

**Focus:** Institutional-quality backtesting, advanced signal generation, multi-strategy portfolios, professional risk management



