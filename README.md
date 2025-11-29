# QuantTrading

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

### 1. Simple Backtest with New Signals

```python
from core.portfolio.portfolio_manager_v2 import PortfolioManagerV2
from core.multi_asset_loader import load_assets
from signals.trend_following_long_short import TrendFollowingLongShort

# Load data
prices = load_assets(['ES'], start_date='2015-01-01', end_date='2024-12-31')

# Generate signals
signal_gen = TrendFollowingLongShort(
    fast_period=20,
    slow_period=100,
    momentum_threshold=0.02
)
signals = {'ES': signal_gen.generate(prices['ES'])}

# Run backtest
pm = PortfolioManagerV2(
    initial_capital=100000,
    risk_per_trade=0.02,
    max_position_size=0.30,
    transaction_cost_bps=3.0,
    slippage_bps=2.0
)

result = pm.run_backtest(signals, prices)
result.print_summary()
result.plot_equity_curve()
```

### 2. Multi-Strategy Portfolio with Ensemble

```python
from signals.ensemble import AdaptiveEnsemble
from signals.momentum import MomentumSignalV2

# Configure strategies
strategies = [
    {
        'name': 'Adaptive_Ensemble',
        'signal_generator': AdaptiveEnsemble(
            strategies=[
                ('momentum', MomentumSignalV2(lookback=60), 0.5),
                ('trend_ls', TrendFollowingLongShort(), 0.5)
            ],
            method='adaptive',
            adaptive_lookback=60,
            signal_threshold=0.3
        ),
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

# Run all strategies and generate reports
from core.multi_strategy_reporter import MultiStrategyReporter
from core.risk_dashboard import RiskDashboard

reporter = MultiStrategyReporter()
risk_dash = RiskDashboard()

# Generate HTML reports
performance_html = reporter.generate_report(
    strategy_results=results,
    combined_equity=combined_equity,
    benchmark_data=spy_data,
    title="Multi-Strategy Performance"
)

risk_html = risk_dash.generate_dashboard(
    strategy_results=results,
    combined_equity=combined_equity,
    benchmark_data=spy_data
)
```

### 3. Explore Research Notebooks

```bash
jupyter lab notebooks/multi_strategy_with_ensemble.ipynb
```

Comprehensive notebooks demonstrating:
- Multi-strategy portfolio construction
- Adaptive ensemble configuration
- Benchmark comparison (SPY)
- Performance attribution
- Risk analysis
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
│   │   └── backtest_result.py            # Results container
│   │
│   ├── multi_asset_loader.py             # Futures data loader
│   ├── multi_strategy_reporter.py        # Performance reports (HTML)
│   ├── risk_dashboard.py                 # Risk analysis dashboard
│   ├── benchmark.py                      # SPY comparison tools
│   └── optimizer.py                      # Walk-forward optimization
│
├── signals/                               # Trading strategies
│   ├── base.py                           # SignalModel abstract base
│   ├── momentum.py                       # Momentum strategies
│   ├── mean_reversion.py                 # Counter-trend strategies
│   ├── trend_following_long_short.py     # Long-short trend (NEW)
│   ├── ensemble.py                       # Adaptive ensembles (NEW)
│   └── README_NEW_SIGNALS.md             # Signal documentation
│
├── notebooks/                             # Research notebooks
│   ├── multi_strategy_with_ensemble.ipynb # Main demo (comprehensive)
│   ├── test_new_signals.ipynb            # Signal testing
│   └── backtest_momentum.ipynb           # Single-strategy research
│
├── reports/                               # Generated HTML reports
│   ├── ensemble_performance.html         # Portfolio performance
│   └── ensemble_risk_dashboard.html      # Risk analysis
│
├── Dataset/                               # Market data
│   ├── spx_data.csv                      # S&P 500 futures (ES)
│   ├── nq_data.csv                       # NASDAQ futures (NQ)
│   └── gc_data.csv                       # Gold futures (GC)
│
├── utils/                                 # Utilities
│   ├── logger.py                         # Logging
│   └── metrics.py                        # Performance metrics
│
├── backtest/                              # Legacy backtest engine
├── live/                                  # Paper trading
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
- pandas, numpy
- matplotlib, plotly
- scipy (for statistical analysis)
- jupyter/jupyterlab (for notebooks)

### Setup
```bash
# Clone repository
git clone https://github.com/Gadangadang/QuantTrading.git
cd QuantTrading

# Create environment using conda (recommended)
conda env create -f environment.yml
conda activate quant_trading

# Or use pip
pip install pandas numpy matplotlib plotly scipy jupyterlab
```

## 🧪 Testing

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



