# Master TODO

Consolidated from all project documentation. Items are ordered by priority.

## Critical — Must Fix

- [ ] Fix deprecated Pandas `method='ffill'`/`method='bfill'` calls (breaks on pandas >= 2.0)
- [ ] Remove debug print statements from `core/multi_strategy_reporter.py`
- [ ] Fix `.gitignore` formatting (spaces in glob patterns)
- [ ] Remove stale files (`optimizer.py.backup`, `optimizer.py.old`, committed `coverage.xml`)
- [ ] Fix `paper_trading_engine.py` import from deleted archive module

## High — Validation & Credibility

- [ ] Run 2025 out-of-sample test (true OOS since strategies built on 2015-2024)
- [ ] Walk-forward validation on ensemble signals (12-month train / 3-month test)
- [ ] Parameter optimization inside each training window (grid / Bayesian / random search)

## Medium — Infrastructure & Risk

- [ ] Live paper trading infrastructure (`live/run_daily.py`, `live/monitor.py`)
- [ ] Monitoring dashboard (daily P&L, risk metrics, kill switches)
- [ ] Dynamic position sizing (Kelly criterion, volatility targeting)
- [ ] Realistic execution modeling (slippage as function of volume, partial fills)
- [ ] Multi-asset futures migration (ES, NQ, GC, CL, ZN with rollover logic)
- [ ] Sector/asset-class exposure constraints

## Low — Nice to Have

- [ ] Expand asset universe to 10+ liquid futures (RTY, CL, ZN, 6E, VX)
- [ ] Regime detection (bull/bear/sideways classification)
- [ ] Machine learning signal ensembles
- [ ] Real-time data integration
- [ ] Options strategies integration
- [ ] Intraday trading support
- [ ] Email/SMS alerts on large drawdowns

## Completed

- [x] Core vectorized backtesting system
- [x] 6 institutional signal generators (momentum, mean reversion, ensemble, long-short, hybrid adaptive, energy seasonal)
- [x] Modular V2 portfolio system (Portfolio / RiskManager / ExecutionEngine / BacktestResult)
- [x] Walk-forward validation framework
- [x] Comprehensive HTML reporting with Plotly
- [x] Benchmark comparison (SPY alpha/beta analysis)
- [x] Adaptive ensemble with dynamic weighting
- [x] Trend-following long-short signals
- [x] Risk controls & kill switches
- [x] CI/CD pipeline with GitHub Actions (Python 3.9/3.10/3.11)
- [x] 261 passing tests, Codecov integration
- [x] Asset registry with futures metadata
- [x] Futures contract position sizing
- [x] TAA optimizer (mean-variance, risk-parity, HRP)
