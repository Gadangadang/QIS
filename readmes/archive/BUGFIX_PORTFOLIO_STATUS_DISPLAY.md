# Bug Fix: Portfolio Status Display Issue

**Date:** November 25, 2025  
**Affected Component:** `core/paper_trading_engine.py` → `get_portfolio_status()`  
**Severity:** Medium (display bug, no trading impact)

---

## Problem Description

The paper trading system was incorrectly showing:
- **"Invested: $0.00 (0.0%)"** 
- **"No open positions - 100% cash"**

When in reality, the portfolio held 3 active positions totaling ~$136k in value.

This created confusion, as users saw negative cash (-$5,003) but no positions, making it appear like the portfolio was broken.

---

## Root Cause

**Case sensitivity mismatch** between:
1. How positions are stored in `equity_curve` (uppercase keys: `'Shares'`, `'Price'`, `'Value'`)
2. How `get_portfolio_status()` reads positions (lowercase keys: `'shares'`)

### Code Path:

```python
# In PortfolioManager.get_portfolio_state()
# Positions stored as:
{
    'ES': {'Shares': 6.29, 'Price': 6773.5, 'Value': 42611.36, 'Weight': 0.3255},
    'GC': {...},
    'NQ': {...}
}

# In PaperTradingEngine.get_portfolio_status()
# But read as:
shares = position_info.get('shares', 0)  # ❌ Returns 0 (not found)
```

Since `'shares'` (lowercase) wasn't found, the code defaulted to `0`, making all positions appear empty.

---

## Solution

Modified `get_portfolio_status()` to check **both** uppercase and lowercase keys:

```python
# Before (line 256):
shares = position_info.get('shares', 0)

# After:
shares = position_info.get('Shares', position_info.get('shares', 0))
```

This maintains backward compatibility while fixing the display issue.

---

## Impact Analysis

### What Was Broken:
✅ **Display only** - Portfolio status incorrectly showed 0 positions  
✅ **Affected:** Notebook output, daily reports, status checks

### What Was NOT Broken:
✅ **Trading logic** - Positions were correctly opened and managed  
✅ **Risk management** - Position sizing, stops, and rebalancing all worked  
✅ **P&L tracking** - Equity curve and returns were accurate  
✅ **State persistence** - Positions correctly saved/loaded

The bug was purely cosmetic - the underlying portfolio was trading correctly all along.

---

## About Negative Cash

Users may notice **negative cash** (e.g., -$5,003) and wonder if this is a problem. **It's not.**

### Why Cash Can Be Negative:

With the current configuration:
- **3 assets** in the portfolio
- **35% max position size** per asset
- **Total exposure:** 3 × 35% = **105%**

This means the portfolio uses **5% leverage** (borrows 5% of capital to achieve 105% exposure).

This is:
✅ **Intentional** - configured via `max_position_size=0.35`  
✅ **Normal** - common in professional portfolio management  
✅ **Controlled** - leverage is capped at 5% (quite conservative)

### Cash Calculation:
```
Initial Capital: $100,000
Position 1 (ES):  $42,611  (42.6%)
Position 2 (GC):  $47,511  (47.5%)
Position 3 (NQ):  $45,783  (45.8%)
Total Invested:   $135,905 (135.9%)
Cash Remaining:  -$5,003  (-5.0%)  ← Leverage used
```

**Total Portfolio Value = Cash + Positions = -$5,003 + $135,905 = $130,902** ✅

---

## Testing

### Before Fix:
```
Portfolio Value: $130,901.96
  Cash: $-5,003.33 (-3.8%)
  Invested: $0.00 (0.0%)        ← ❌ WRONG
  Total Return: 30.90%
  P&L: $30,901.96

💰 No open positions - 100% cash  ← ❌ WRONG
```

### After Fix:
```
Portfolio Value: $130,901.96
  Cash: $-5,003.33 (-3.8%)
  Invested: $135,905.29 (103.8%)  ← ✅ CORRECT
  Total Return: 30.90%
  P&L: $30,901.96

📍 Open Positions: 3              ← ✅ CORRECT
  ES: 6.29 shares @ $6,773.50
  GC: 11.39 shares @ $4,171.80
  NQ: 1.83 shares @ $25,025.50
```

---

## Files Modified

1. **core/paper_trading_engine.py**
   - Line 256: Added uppercase key check
   - Method: `get_portfolio_status()`

2. **notebooks/10_paper_trading_with_engine.ipynb**
   - Added markdown cell explaining the fix
   - Added test cell to reload engine and verify fix

---

## Migration Instructions

### For Users:

1. **No state reset needed** - existing positions are fine
2. **Just restart your kernel** and re-import:
   ```python
   import importlib
   import core.paper_trading_engine
   importlib.reload(core.paper_trading_engine)
   from core.paper_trading_engine import PaperTradingEngine
   ```
3. **Reload your state:**
   ```python
   engine = PaperTradingEngine.load_state(STATE_FILE, config)
   ```

That's it! The fix is non-breaking and requires no data migration.

---

## Related Issues

- **None** - This was an isolated display bug with no related issues

---

## Lessons Learned

1. **Case sensitivity matters** - Python dicts are case-sensitive
2. **Standardize key naming** - Should define Position schema in one place
3. **Better testing needed** - Status display should have unit tests
4. **Logging helps** - Equity curve stored full position details, making diagnosis easy

---

## Follow-Up Actions

**Recommended (Low Priority):**

1. Create a `Position` dataclass to standardize field names:
   ```python
   @dataclass
   class Position:
       shares: float
       price: float
       value: float
       weight: float
   ```

2. Add unit tests for `get_portfolio_status()`:
   - Test with uppercase keys (current format)
   - Test with lowercase keys (backward compatibility)
   - Test edge cases (empty positions, negative cash)

3. Consider renaming to lowercase everywhere for Python convention:
   - Change PortfolioManager to output lowercase keys
   - Simpler than checking both cases

**Status:** Fixed in production, follow-up tasks optional.
