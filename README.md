# Quantitative Investment Strategies (QIS)

[![Tests](https://github.com/Gadangadang/QIS/actions/workflows/test.yml/badge.svg)](https://github.com/Gadangadang/QIS/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/Gadangadang/QIS/graph/badge.svg?token=O8O1H8OE9J)](https://codecov.io/gh/Gadangadang/QIS)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A personal research framework for building and testing systematic investment strategies — from signal design and feature engineering to backtesting, walk-forward validation, and performance analysis. Built to apply ML and quant methods to real market data in a rigorous, reproducible way.

> **Portfolio project** — results are for research and framework demonstration only, not investment advice. Work in progress.

## What this is

This started as a structured way to explore cross-asset systematic strategies outside of work. The goal is a full pipeline: ingest data, engineer features, develop signals, validate them honestly (walk-forward, OOS, leakage control), and analyze the output. The TAA module in particular mirrors the kind of quantitative allocation work I do professionally.

## Key Features

| Area | Details |
|------|---------|
| **Testing** | CI across Python 3.9 / 3.10 / 3.11 · Codecov coverage tracking |
| **Architecture** | Modular V2 design — Portfolio · Risk · Execution · Analysis |
| **Signals** | Momentum (V1/V2), Mean Reversion, Hybrid Adaptive, Trend-Following L/S, Ensemble (static + adaptive), Energy Seasonal (V1/V2) |
| **TAA Module** | Feature pipeline (price, macro/FRED, relative value) · ML-enhanced TAA prototype (gradient-boosted trees + regime classifiers) |
| **Validation** | Walk-forward optimization · expanding-window OOS · strict leakage controls via time-split |
| **Risk** | Dynamic sizing (Kelly, ATR, volatility-scaled), stop-loss, drawdown limits, concentration controls |
| **Reporting** | Interactive HTML dashboards (Plotly) · benchmark comparison (alpha/beta vs SPY) · monthly returns heatmap |
| **Assets** | Equities, rates, commodities, FX — futures supported with integer contract sizing |

## Architecture

```
PortfolioManagerV2
├── Portfolio           — position & cash state management
├── RiskManager         — sizing, stops, drawdown monitoring
├── ExecutionEngine     — transaction costs (3 bps) + slippage (2 bps)
└── BacktestResult      — Sharpe, Sortino, Calmar, HTML reports

BacktestOrchestrator    — high-level multi-strategy API (method chaining)
WalkForwardEngine       — rolling train/test optimization with param grid search

core/taa/
├── features/           — price, macro (FRED), relative value feature generators
├── pipeline.py         — end-to-end feature pipeline (data → feature matrix)
└── ml_model.py         — gradient-boosted TAA model + regime classifier prototype
```

Each component is independently testable with clean interfaces.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

### Single strategy

```python
from core.multi_asset_loader import load_assets
from core.portfolio.portfolio_manager_v2 import PortfolioManagerV2
from signals.momentum import MomentumSignalV2

prices = load_assets(['ES', 'GC'], start_date='2018-01-01')

signal = MomentumSignalV2(lookback=60, entry_z=2.0, exit_z=0.5, sma_period=200)
signals = {ticker: signal.generate(df) for ticker, df in prices.items()}

pm = PortfolioManagerV2(initial_capital=100_000, risk_per_trade=0.02)
result = pm.run_backtest(signals, prices)
result.print_summary()
```

### Multi-strategy via orchestrator

```python
from core.backtest_orchestrator import BacktestOrchestrator
from signals.momentum import MomentumSignalV2
from signals.mean_reversion import MeanReversionSignal

(BacktestOrchestrator()
    .load_data(['ES', 'NQ', 'GC'], start_date='2018-01-01')
    .load_benchmark('SPY')
    .add_strategy('Momentum',      MomentumSignalV2(lookback=60),     ['ES', 'NQ'], capital=50_000)
    .add_strategy('MeanReversion', MeanReversionSignal(window=20),    ['GC'],       capital=25_000)
    .generate_signals()
    .run_backtests()
    .print_summary())
```

### Walk-forward optimization

```python
from core.walk_forward_optimizer import WalkForwardOptimizer
from signals.momentum import MomentumSignalV2

engine = WalkForwardOptimizer(
    signal_class=MomentumSignalV2,
    param_grid={'lookback': [60, 90, 120], 'sma_period': [150, 200]},
    tickers=['ES', 'GC', 'NQ'],
    start_date='2010-01-01',
    train_years=3,
    test_years=1
)
results = engine.run()
print(f"Best params: {results['best_params']}")
print(f"Avg OOS Sharpe: {results['oos_sharpe_mean']:.2f}")
```

### Sample output

```
=====================================================================
BACKTEST RESULTS SUMMARY
======================================================================

PERFORMANCE METRICS
-----------------------------------
Total Return              +34.21%
CAGR                       +8.14%
Sharpe Ratio                  1.23
Sortino Ratio                 1.71
Max Drawdown              -12.44%
Calmar Ratio                  0.65

BENCHMARK COMPARISON (SPY)
-----------------------------------
Alpha (Annual)             +3.12%
Beta (Full Period)            0.41
```
*(Sample output — actual results vary by asset, period, and parameters.)*

## Project Structure

```
core/
├── portfolio/           # V2 modular system (manager, risk, execution, sizing)
├── data/
│   ├── collectors/      # Yahoo, FactSet, FRED data collectors
│   └── processors/      # Price normalization and alignment
├── taa/
│   ├── features/        # Price, macro (FRED), relative value feature generators
│   └── pipeline.py      # End-to-end feature pipeline
├── futures/             # Contract rollover handling
├── backtest_orchestrator.py
├── walk_forward_optimizer.py
├── benchmark.py
├── multi_asset_signal.py
├── multi_strategy_reporter.py
└── risk_dashboard.py

signals/                 # Signal generators
├── momentum.py              # MomentumSignal, MomentumSignalV2
├── mean_reversion.py        # MeanReversionSignal
├── hybrid_adaptive.py       # HybridAdaptiveSignal (regime-switching)
├── trend_following_long_short.py  # TrendFollowingLongShort (L/S)
├── ensemble.py              # EnsembleSignal, AdaptiveEnsemble
├── energy_seasonal.py       # EnergySeasonalSignal (V1)
└── energy_seasonal_v2.py    # EnergySeasonalBalanced, EnergySeasonalAggressive

utils/                   # Plotting, formatting, param loading
tests/                   # pytest test suite
notebooks/               # Research & demo notebooks
```

## Testing & CI

```bash
pytest tests/ -v --cov=core --cov=signals --cov-report=term-missing
```

GitHub Actions runs on every push — tests across Python 3.9 / 3.10 / 3.11, Codecov upload, and linting (black, flake8, isort). See `.github/workflows/test.yml`.

## Documentation

- [Signal Development Guide](signals/README.md)
- [New Signals (Trend-Following L/S & Ensemble)](signals/README_NEW_SIGNALS.md)
- [Backtest Engine & Walk-Forward](core/BACKTEST_README.md)
- [Roadmap & TODO](readmes/TODO.md)

## License

Private research project. All rights reserved.

---

**Built with** Python · Pandas · NumPy · SciPy · PyTorch · Plotly