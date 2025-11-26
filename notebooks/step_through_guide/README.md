# Step-Through Guide

## Purpose
This folder contains educational notebooks that teach you how the backtesting system works from the ground up. Each notebook focuses on one component with detailed explanations, print statements, and source code references.

## Learning Path

### 1️⃣ **01_basic_backtest.ipynb**
**What you'll learn:**
- How a single-asset backtest works step-by-step
- Position initialization and updates
- Cash management
- Trade execution and transaction costs
- Equity curve generation

**Time:** 30 minutes

---

### 2️⃣ **02_portfolio_manager_deep_dive.ipynb**
**What you'll learn:**
- PortfolioManager class architecture
- Multi-asset position management
- Signal-based entry/exit logic
- Rebalancing mechanics
- How positions, cash, and equity track together

**Time:** 45 minutes

---

### 3️⃣ **03_risk_manager_explained.ipynb**
**What you'll learn:**
- RiskManager integration with portfolio
- Position sizing algorithms (vol-adjusted, Kelly, fixed)
- Correlation monitoring
- Drawdown stops and violation tracking
- Trade validation logic

**Time:** 30 minutes

---

### 4️⃣ **04_full_system_integration.ipynb**
**What you'll learn:**
- How all components work together
- Data flow from signals → risk checks → portfolio → trades
- Multi-asset backtest with full risk management
- Complete end-to-end example

**Time:** 45 minutes

---

### 5️⃣ **05_walk_forward_framework.ipynb**
**What you'll learn:**
- Walk-forward validation methodology
- Train/test split mechanics
- Parameter isolation (no lookahead bias)
- Out-of-sample performance tracking
- Fold-by-fold analysis

**Time:** 45 minutes

---

## How to Use These Notebooks

1. **Run cells sequentially** - Don't skip ahead
2. **Read the print output carefully** - Shows exactly what's happening
3. **Refer to source code** - Open the corresponding .py files side-by-side
4. **Experiment** - Change parameters and re-run to see effects
5. **Take notes** - Jot down questions or insights as you go

## Key Files Referenced

- `core/portfolio_manager.py` - Main portfolio orchestration
- `core/risk_manager.py` - Risk limits and position sizing
- `core/multi_asset_loader.py` - Data loading utilities
- `signals/momentum.py` - Example signal implementation
- `core/paper_trading_engine.py` - State persistence wrapper

## Tips

- **Print statements everywhere** - These notebooks use extensive print() to show state changes
- **Small datasets** - We use short time periods (1-2 years) so you can see all the data
- **Visual diagrams** - ASCII art shows data flow at key points
- **Assertions** - We verify expected behavior with assert statements

## After Completing This Guide

You should be able to:
- ✅ Explain how money flows through the system
- ✅ Understand every line of a backtest
- ✅ Debug issues in portfolio management
- ✅ Add new features confidently
- ✅ Design walk-forward tests correctly
- ✅ Evaluate strategies rigorously

---

**Ready to start? Open `01_basic_backtest.ipynb`** 🚀
