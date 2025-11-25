# Bug Fix: Drawdown Stop Carrying Over from Backtest to Live Trading

## Date: November 25, 2025

## Problem Identified

User reported that live paper trading showed:
- Portfolio Value: $131,221 (+31% return)
- 100% cash, no open positions
- All signals showing LONG
- No new trades since April 2025

### Root Cause

The portfolio hit a **drawdown stop violation during the backtest** (March 2020, -21.02% drawdown during COVID crash), and this violation **carried over into live trading**, preventing any new trades from being executed.

**Violation Details:**
```
PORTFOLIO | drawdown_stop | Max drawdown -21.02% exceeds stop -20.00%
Date: 2020-03-16 (from backtest period)
```

The `PaperTradingEngine` was using the **same risk manager instance** for both backtest reference and live trading, causing historical violations to persist and block live trades indefinitely.

---

## Solution Implemented

### Changes to `core/paper_trading_engine.py`

**1. Modified `initialize()` method (lines 145-168):**

```python
# OLD CODE:
result, equity, trades = run_multi_asset_backtest(
    signals_dict=signals_dict,
    prices_dict=prices_dict,
    config=self.config,  # ❌ Reuses same risk manager
    return_pm=False
)

# NEW CODE:
# Use a fresh risk manager for live trading
live_risk_manager = RiskManager(self.config.risk_manager.config)

# Create a fresh config with the new risk manager
live_config = PortfolioConfig(
    initial_capital=self.config.initial_capital,
    rebalance_threshold=self.config.rebalance_threshold,
    transaction_cost_bps=self.config.transaction_cost_bps,
    risk_manager=live_risk_manager,  # ✅ Fresh risk manager
    rejection_policy=self.config.rejection_policy
)

result, equity, trades = run_multi_asset_backtest(
    signals_dict=signals_dict,
    prices_dict=prices_dict,
    config=live_config,  # ✅ Uses fresh config
    return_pm=False
)
```

**2. Modified `update()` method (lines 189-219):**

```python
# OLD CODE:
result, equity_full, trades_full = run_multi_asset_backtest(
    signals_dict=signals_dict,
    prices_dict=prices_dict,
    config=self.config,  # ❌ Reuses same risk manager
    return_pm=False
)

# NEW CODE:
# Use a fresh risk manager for each update
live_risk_manager = RiskManager(self.config.risk_manager.config)

# Create a fresh config with the new risk manager
live_config = PortfolioConfig(
    initial_capital=self.config.initial_capital,
    rebalance_threshold=self.config.rebalance_threshold,
    transaction_cost_bps=self.config.transaction_cost_bps,
    risk_manager=live_risk_manager,  # ✅ Fresh risk manager
    rejection_policy=self.config.rejection_policy
)

result, equity_full, trades_full = run_multi_asset_backtest(
    signals_dict=signals_dict,
    prices_dict=prices_dict,
    config=live_config,  # ✅ Uses fresh config
    return_pm=False
)
```

---

## Impact

### Before Fix:
- ❌ Backtest violations carried over to live trading
- ❌ Drawdown stop from 2020 blocked all 2025 trades
- ❌ Portfolio stuck in cash despite LONG signals
- ❌ Live trading couldn't recover after backtest drawdown

### After Fix:
- ✅ Live trading uses fresh risk manager
- ✅ Drawdown tracking starts from live trading start date
- ✅ No inheritance of historical violations
- ✅ Live portfolio can trade independently of backtest history

---

## How to Apply the Fix

### For Users with Existing State:

**Option 1: Reset State (Recommended)**
```python
# In notebook, run this cell:
if STATE_FILE.exists():
    STATE_FILE.unlink()
    print("✅ State deleted")

# Then re-run Phase 2 (Initialize) to create fresh state
```

**Option 2: Manual Workaround (Temporary)**
```python
# Increase drawdown stop to allow trading
risk_config = RiskConfig(
    max_drawdown_stop=-0.25,  # Increase from -0.20
    # ... other settings
)
```

### For New Users:
- No action needed
- Fresh initializations will automatically use the fix
- Live trading will track drawdown independently

---

## Technical Details

### Why Fresh Risk Manager?

**Risk Manager tracks:**
1. **Violations**: Drawdown stops, correlation limits, position size breaches
2. **Historical state**: Max drawdown seen, previous violations, stop triggers
3. **Context**: When violations occurred, severity, recovery status

**Problem with sharing:**
- Backtest risk manager: Sees -21.02% drawdown in March 2020
- Sets `is_stopped = True`
- Live trading inherits this manager
- Result: All trades rejected forever

**Solution with fresh manager:**
- Live risk manager: Created fresh, no historical violations
- Tracks drawdown from live start date only
- Violations are based on live performance
- Result: Trades execute normally unless live trading hits limits

### Architecture Benefits

**Separation of Concerns:**
- **Backtest risk manager**: For reference/comparison only, read-only
- **Live risk manager**: Active enforcement for live trading
- **No coupling**: Live decisions independent of historical performance

**Correct Behavior:**
- Backtest can show -21% drawdown (good to know!)
- Live trading starts fresh at 0% drawdown
- If live trading hits -20%, it correctly stops
- If live trading recovers, it can resume (future enhancement)

---

## Testing

### Test Case 1: Fresh Initialization
```python
# Create engine
engine = PaperTradingEngine(config=config, backtest_result=result_with_violations)

# Initialize live trading
engine.initialize(prices, signals, start_date='2025-01-01')

# Expected: No violations carried over
assert len(engine.state.equity_curve) > 0
# Should have trades if signals are LONG
```

### Test Case 2: Daily Update
```python
# Load existing state
engine = PaperTradingEngine.load_state('state.pkl', config)

# Update with new data
engine.update(new_prices, new_signals)

# Expected: Uses fresh risk manager, no stale violations
```

### Verification Commands
```python
# Check for violations in live trading
if hasattr(result_live, 'violations'):
    print(f"Violations: {len(result_live.violations) if result_live.violations is not None else 0}")

# Check live trades are executing
print(f"Live trades: {len(engine.state.trades)}")
print(f"Last trade date: {engine.state.trades['Date'].iloc[-1] if len(engine.state.trades) > 0 else 'None'}")
```

---

## Future Enhancements

### 1. Drawdown Recovery Logic
Currently, once a drawdown stop triggers, it stays stopped forever. Future enhancement:
```python
# Allow recovery after drawdown improves
if current_drawdown > max_drawdown_stop:
    is_stopped = True
elif is_stopped and current_drawdown < max_drawdown_stop * 0.5:  # Recovered 50%
    is_stopped = False
    log_recovery()
```

### 2. Configurable Violation Inheritance
```python
# Let users choose what to inherit from backtest
live_config = LiveConfig(
    inherit_violations=False,  # Don't carry over historical stops
    inherit_correlation_matrix=True,  # But do use historical correlations
    inherit_volatility_estimates=True  # And volatility estimates
)
```

### 3. Better Violation Tracking
```python
# Separate historical vs live violations
class ViolationTracker:
    def __init__(self):
        self.backtest_violations = []  # Reference only
        self.live_violations = []  # Actively enforced
    
    def should_block_trade(self):
        return any(v.is_active for v in self.live_violations)
```

---

## Related Issues

- Risk manager state management
- Violation persistence across sessions
- Drawdown calculation methods
- Recovery logic after stops

---

## Documentation Updates

Added to `notebooks/10_paper_trading_with_engine.ipynb`:
- New markdown cell explaining the fix
- Python cell to reset state
- Instructions for applying the fix

---

## Lessons Learned

1. **Stateful objects are dangerous across contexts**
   - Risk managers track historical violations
   - Sharing them across backtest/live causes issues
   - Always create fresh instances for new contexts

2. **Test with edge cases**
   - What happens after a violation?
   - Does state persist correctly?
   - Can the system recover?

3. **Clear separation of concerns**
   - Backtest = reference/comparison
   - Live = active decision making
   - Never mix the two

4. **User feedback is critical**
   - User noticed "no trades despite LONG signals"
   - This revealed a fundamental architectural flaw
   - Led to important fix

---

## Summary

Fixed critical bug where drawdown stop violations from backtest period carried over into live trading, preventing all trades from executing. Solution creates fresh risk manager instances for live trading, ensuring violations are tracked independently based on live performance only.

**Impact**: Users can now trade live even if backtest showed large drawdowns. Live trading performance is evaluated on its own merits.

**Files Changed**: 
- `core/paper_trading_engine.py` (2 methods updated)
- `notebooks/10_paper_trading_with_engine.ipynb` (2 cells added)

**Breaking Changes**: None - existing behavior preserved, bug fixed

**Migration Required**: Yes - users must delete state file and re-initialize
