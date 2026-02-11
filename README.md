# Quantitative Trading System

[![Tests](https://github.com/Gadangadang/QIS/actions/workflows/test.yml/badge.svg)](https://github.com/Gadangadang/QIS/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/Gadangadang/QIS/graph/badge.svg?token=O8O1H8OE9J)](https://codecov.io/gh/Gadangadang/QIS)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Production-grade multi-strategy backtesting framework built in Python. Demonstrates software engineering best practices applied to quantitative finance — modular architecture, comprehensive testing, CI/CD, and realistic cost modeling.

> **Portfolio project** — results are for framework demonstration only, not investment advice.

## Key Highlights

| Area | Details |
|------|---------|
| **Testing** | 820 passing tests · 70% coverage · CI across Python 3.9 / 3.10 / 3.11 · Codecov |
| **Architecture** | Modular V2 design — Portfolio · Risk · Execution · Analysis |
| **Signals** | 8 strategies (momentum, mean-reversion, trend-following L/S, adaptive ensemble) |
| **Risk** | Dynamic sizing (Kelly, ATR), stop-loss, drawdown limits, concentration controls |
| **Reporting** | Interactive HTML dashboards (Plotly), benchmark comparison (alpha/beta vs SPY) |
| **Assets** | Equities & commodity futures with integer contract sizing |

## Architecture

```
PortfolioManagerV2
├── Portfolio          — position & cash state management
├── RiskManager        — sizing, stops, drawdown monitoring
├── ExecutionEngine    — transaction costs (3 bps) + slippage (2 bps)
└── BacktestResult     — Sharpe, Sortino, Calmar, HTML reports
```

Each component is independently testable with clean interfaces.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

```python
from core.portfolio.portfolio_manager_v2 import PortfolioManagerV2
from signals.momentum import MomentumSignalV2

# Generate signals & backtest
signal = MomentumSignalV2(lookback=60, entry_threshold=0.02)
signals = {'ES': signal.generate(prices['ES'])}

pm = PortfolioManagerV2(initial_capital=100_000, risk_per_trade=0.02)
result = pm.run_backtest(signals, prices)
result.print_summary()
```

Multi-strategy portfolios are supported via `BacktestOrchestrator` — see [Backtest Engine docs](core/BACKTEST_README.md).

## Project Structure

```
core/
├── portfolio/          # V2 modular system (manager, risk, execution, sizing)
├── futures/            # Rollover handling
├── taa/                # Tactical asset allocation optimizer
├── backtest_orchestrator.py
├── benchmark.py
├── multi_strategy_reporter.py
└── risk_dashboard.py

signals/                # Signal generators (base, momentum, mean-reversion,
                        #   trend-following L/S, ensemble, hybrid, energy-seasonal)

utils/                  # Plotting, formatting, param loading
tests/                  # 820 tests (pytest)
notebooks/              # Research & demo notebooks
```

## Testing & CI

```bash
pytest tests/ -v --cov=core --cov=signals --cov-report=term-missing
```

GitHub Actions runs on every push — tests on three Python versions, Codecov upload, and linting (black, flake8, isort). See `.github/workflows/test.yml`.

## Documentation

- [Signal Development Guide](signals/README.md)
- [New Signals (Trend-Following L/S & Ensemble)](signals/README_NEW_SIGNALS.md)
- [Backtest Engine](core/BACKTEST_README.md)
- [Roadmap & TODO](readmes/TODO.md)

## License

Private research project. All rights reserved.

---

**Built with** Python · Pandas · NumPy · SciPy · Plotly

