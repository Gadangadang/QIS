# QuantTrading

A modular quantitative trading system for paper trading with walk-forward validation, comprehensive reporting, and notebook-first research workflow.

## 🎯 Goal

Build a reproducible quantitative strategy system to demonstrate alpha generation through backtesting and daily execution, with a focus on clean code, rigorous validation, and easy experimentation.

**Current Status:** Paper trading only (DNB Asset Management compliance restrictions prevent live broker connections).

## 📊 Quick Start

### 1. Run a Test Backtest
```bash
python3 test_new_structure.py
```

This runs a quick backtest and generates an HTML report at `logs/test_new_structure/report.html`.

### 2. Open the Research Notebook
```bash
jupyter lab notebooks/03_backtest_momentum.ipynb
```

Configure your strategy with a simple Python dict and run interactive backtests with auto-generated reports.

### 3. View Results
```bash
open logs/test_new_structure/report.html
```

Interactive HTML report with:
- Performance metrics (Sharpe, Sortino, Calmar, CAGR, MaxDD)
- Trade statistics (win rate, profit factor, avg win/loss)
- Regime analysis (correlation to market, up/down day performance)
- Interactive charts (equity curve, drawdown, returns distribution)

## 🏗️ Project Structure

```
QuantTrading/
├── core/                       # Execution engine
│   ├── backtest_engine.py     # Walk-forward validation
│   └── paper_trader.py        # Trading simulator with risk controls
│
├── analysis/                   # Diagnostics & reporting
│   ├── report.py              # BacktestReport class (HTML reports)
│   └── metrics.py             # Performance metrics (Sharpe, Sortino, etc.)
│
├── signals/                    # Signal models
│   ├── base.py                # Abstract SignalModel base class
│   ├── momentum.py            # Momentum strategy
│   ├── mean_reversion.py      # Mean reversion strategy
│   └── ensemble.py            # Ensemble strategies
│
├── notebooks/                  # Research notebooks
│   └── 03_backtest_momentum.ipynb  # Main research template
│
├── scripts/                    # Utility scripts
│   ├── run_daily.py           # Daily paper trading runner
│   ├── sanity_check_signal.py # Signal validation
│   └── test_stop_loss.py      # Stop-loss testing
│
├── Dataset/                    # Data files
│   └── spx_full_1990_2025.csv # S&P 500 daily data (1990-2025)
│
├── logs/                       # Output directory
│   └── [experiment_name]/     # Auto-generated per run
│       ├── report.html        # Interactive HTML report
│       ├── stitched_equity.csv
│       ├── combined_returns.csv
│       └── trades_fold_*.csv
│
└── archive/                    # Old code (can be deleted after 1 week)
    └── old_structure/
```

## 🚀 New Notebook-First Workflow

### Before (Old CLI - Deprecated)
```bash
python3 -m backtest.runner walkforward --signal momentum --train-frac 0.6 \
  --test-frac 0.2 --lookback 250 --stop-loss 0.1 --stop-mode low \
  --max-pos 0.2 --save-dir logs/test --transaction-cost 3.0 ...
```

### After (New Notebook Workflow)
```python
# In Jupyter notebook - all config visible and easy to modify
config = {
    'signal_factory': lambda: MomentumSignal(lookback=120, threshold=0.02),
    'df': df,
    'train_size': int(len(df) * 0.6),
    'test_size': int(len(df) * 0.2),
    'lookback': 250,
    'stop_loss_pct': 0.10,
    'transaction_cost': 3.0,
    'save_dir': '../logs/momentum_v1',
}

# Run and automatically get comprehensive report
results = run_walk_forward(**config)

# Interactive exploration
report = BacktestReport(results)
report.summary()
report.worst_days(10)
report.plot_equity().show()
```

## 📈 Key Features

### 1. Walk-Forward Validation
- Anchored walk-forward with configurable train/test splits
- No lookahead bias (uses `Position.shift(1)` for execution)
- Proper compounding across folds
- Per-fold trade tracking with fold numbers

### 2. Comprehensive Risk Controls
- **Stop-loss:** Per-trade or global percentage stops
- **Take-profit:** Optional profit targets
- **Max hold:** Maximum days per trade
- **Stop modes:** `close` (EOD), `low` (intraday), `open` (intraday)
- **Position sizing:** Percentage of capital per trade
- **Transaction costs:** Basis points per trade

### 3. Comprehensive Reporting (BacktestReport Class)
Automatically generated for every backtest with `save_dir` specified:

**Performance Metrics:**
- Sharpe Ratio
- Sortino Ratio (downside deviation only)
- Calmar Ratio (CAGR / abs(MaxDD))
- CAGR (annualized)
- Maximum Drawdown

**Trade Statistics:**
- Number of trades
- Win rate
- Profit factor (gross profit / gross loss)
- Average win/loss
- Largest win/loss

**Regime Analysis:**
- Correlation to market
- Performance on up vs. down days

**Interactive Charts:**
- Equity curve with drawdown shading
- Daily returns distribution
- Trade PnL histogram
- Cumulative returns
- Monthly returns heatmap

### 4. Signal Framework
All signals inherit from `SignalModel` base class:

```python
class SignalModel(ABC):
    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with Position column {-1, 0, 1}"""
        pass
```

**Available Signals:**
- `MomentumSignal(lookback, threshold)` - Trend following
- `MeanReversionSignal(window, entry_z, exit_z)` - Mean reversion
- `EnsembleSignal()` - Multi-timeframe momentum ensemble
- `EnsembleSignalNew()` - Mean reversion ensemble with trend filter

## 📚 Usage Examples

### Basic Backtest in Notebook
```python
from core.backtest_engine import run_walk_forward
from signals.momentum import MomentumSignal
from analysis.report import BacktestReport

# Load data
df = pd.read_csv('../Dataset/spx_full_1990_2025.csv', index_col=0, parse_dates=True)

# Configure
config = {
    'signal_factory': lambda: MomentumSignal(lookback=120, threshold=0.02),
    'df': df,
    'train_size': int(len(df) * 0.6),
    'test_size': int(len(df) * 0.2),
    'lookback': 250,
    'transaction_cost': 3.0,
    'stop_loss_pct': 0.10,
    'save_dir': '../logs/momentum_test',
}

# Run (auto-generates report)
results = run_walk_forward(**config)
```

### Analyze Results
```python
# Create report object
report = BacktestReport(results)

# Print summary
report.summary()

# Get worst days
worst_days = report.worst_days(10)

# Get worst trades
worst_trades = report.worst_trades(10)

# Interactive plots (if plotly installed)
fig = report.plot_equity()
fig.show()

# Access raw data for custom analysis
results['trades']  # All trades with fold numbers
results['combined_returns']  # Daily strategy returns
results['stitched_equity']  # Portfolio value over time
```

### Try Different Signals
```python
from signals.mean_reversion import MeanReversionSignal
from signals.ensemble import EnsembleSignalNew

# Mean reversion
config['signal_factory'] = lambda: MeanReversionSignal(window=20, entry_z=2.0)

# Ensemble
config['signal_factory'] = lambda: EnsembleSignalNew()
```

## 🔧 Installation

### Requirements
- Python 3.9+
- pandas
- numpy
- matplotlib
- plotly (optional, for interactive HTML reports)
- jupyter/jupyterlab (for notebooks)

### Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install pandas numpy matplotlib plotly jupyterlab

# Or use conda
conda env create -f environment.yml
conda activate quant_trading
```

## 📖 Documentation

- **[RESTRUCTURE_SUMMARY.md](RESTRUCTURE_SUMMARY.md)** - Comprehensive guide to new structure
- **[notebooks/03_backtest_momentum.ipynb](notebooks/03_backtest_momentum.ipynb)** - Interactive research template
- **[archive/ARCHIVE_INFO.md](archive/ARCHIVE_INFO.md)** - Info about archived old code

## 🧪 Testing

Run the comprehensive test suite:
```bash
python3 test_new_structure.py
```

This verifies:
- ✅ All imports working
- ✅ Walk-forward validation running
- ✅ Trades being collected correctly
- ✅ Metrics calculated properly
- ✅ HTML reports generating
- ✅ All output files created

## 🛠️ Development Workflow

1. **Research:** Use notebooks for experimentation
2. **Parameter tuning:** Modify config dict and re-run
3. **Signal development:** Inherit from `SignalModel` in `signals/`
4. **Production:** Use `scripts/run_daily.py` for automated execution

## 📊 Current Performance

Test backtest results (momentum strategy):
- Period: 2004-04-21 to 2025-11-17
- CAGR: -40.12% ⚠️ (needs optimization)
- Sharpe: -0.330
- Max Drawdown: -99.40%
- Trades: 238 (9.2% win rate)

**Note:** Negative performance indicates need for parameter optimization - this is expected for initial testing.

## 🔜 Next Steps

1. **Parameter Optimization** - Grid search on training windows
2. **Volatility Targeting** - Position sizing based on realized vol
3. **Short Borrow Costs** - Add to transaction cost model
4. **Multi-Asset Support** - Extend to futures (ES, NQ, GC)
5. **Portfolio Management** - Risk budgeting across strategies

## 📝 Migration Notes

**Old CLI Deprecated:** The command-line interface (`backtest/runner.py`) has been replaced by the notebook workflow. See `archive/ARCHIVE_INFO.md` for recovery of old code if needed.

**Import Changes:**
- `from backtest.backtest_engine import ...` → `from core.backtest_engine import ...`
- `from live.paper_trader import ...` → `from core.paper_trader import ...`
- `from utils.metrics import ...` → `from analysis.metrics import ...`

## 🤝 Contributing

This is a personal research project for demonstrating quant finance skills. Not currently accepting external contributions.

## 📄 License

Private research project. Not licensed for public use.

---

**Built with:** Python 3.9 | pandas | numpy | matplotlib | plotly | Jupyter

**Generated with:** 🤖 [Claude Code](https://claude.com/claude-code)
