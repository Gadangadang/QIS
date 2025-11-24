# Portfolio Manager Optimization Analysis

**Date:** November 24, 2025  
**File:** `core/portfolio_manager.py`  
**Current Status:** Serial bar-by-bar processing (~566 lines)

---

## 🔴 CRITICAL BOTTLENECKS

### 1. **Main Loop in `run_multi_asset_backtest()` (Lines 490-511)**

**Current Code:**
```python
# Run through all dates
for date in all_dates[1:]:  # 🔴 SERIAL LOOP - MAJOR BOTTLENECK
    # Get current prices and signals
    current_prices = {ticker: df.loc[date, 'Close'] 
                     for ticker, df in prices_dict.items()}
    current_signals = {ticker: df.loc[date, 'Signal'] 
                      for ticker, df in signals_dict.items()}
    
    # Update position values with current prices
    pm.update_positions(current_prices)  # Called ~2500 times for 10 years
    
    # Check if signals changed
    pm.update_signals(current_signals, current_prices, date)
    
    # Update again after signal changes
    pm.update_positions(current_prices)
    
    # Check rebalancing
    if pm.check_rebalance_needed(current_signals):
        pm.rebalance(current_prices, current_signals, date)
        pm.update_positions(current_prices)  # Third time!
    
    # Record state
    pm.equity_curve.append(pm.get_portfolio_state(date))  # Append to list
```

**Issues:**
- **2,500+ iterations** for 10-year backtest (one per trading day)
- `update_positions()` called **2-3 times per day** (unnecessary)
- Dictionary lookups in tight loop (`df.loc[date, 'Close']`)
- State stored in lists/dicts (inefficient for vectorization)
- Multiple passes through data

**Suggested Fix:**
```python
# VECTORIZED APPROACH - Process all dates at once

def run_multi_asset_backtest_vectorized(signals_df, prices_df, config):
    """
    Vectorized backtest - processes all dates simultaneously.
    
    Args:
        signals_df: DataFrame with columns = tickers, index = dates
        prices_df: DataFrame with columns = tickers, index = dates
    """
    # 1. Align all data upfront (one-time operation)
    signals_df = signals_df.reindex(prices_df.index, method='ffill')
    
    # 2. Calculate signal changes vectorized (no loop)
    signal_changes = signals_df.diff().fillna(0)
    entry_dates = signal_changes != 0
    
    # 3. Calculate position sizes vectorized
    # Equal weight among active signals
    n_active = (signals_df != 0).sum(axis=1)  # Count active signals per day
    weights = signals_df.div(n_active, axis=0).fillna(0)  # Equal weight
    
    # 4. Calculate position values vectorized
    shares = (config.initial_capital * weights / prices_df).fillna(0)
    position_values = shares * prices_df
    
    # 5. Calculate rebalancing dates
    if config.rebalance_frequency == 'monthly':
        rebal_dates = prices_df.resample('M').last().index
    elif config.rebalance_frequency == 'weekly':
        rebal_dates = prices_df.resample('W').last().index
    else:
        rebal_dates = []
    
    # 6. Apply rebalancing at specific dates
    for rebal_date in rebal_dates:
        # Vectorized rebalancing logic here
        pass
    
    # 7. Calculate equity curve vectorized
    portfolio_value = position_values.sum(axis=1)
    cash = config.initial_capital - position_values.sum(axis=1)
    total_equity = portfolio_value + cash
    
    # 8. Calculate transaction costs vectorized
    trades = shares.diff().fillna(shares)  # All trades at once
    trade_values = (trades * prices_df).abs()
    transaction_costs = trade_values * (config.transaction_cost_bps / 10000)
    
    # Adjust equity for costs
    cumulative_costs = transaction_costs.sum(axis=1).cumsum()
    total_equity -= cumulative_costs
    
    return total_equity, trades
```

**Performance Gain:** ~50-100x faster (0.1 seconds vs 5-10 seconds)

---

### 2. **`update_positions()` Method (Lines 87-109)**

**Current Code:**
```python
def update_positions(self, prices: Dict[str, float]):
    """Update position values based on current prices."""
    for ticker, price in prices.items():  # 🔴 LOOP PER TICKER PER DAY
        if ticker in self.positions:
            pos = self.positions[ticker]
            pos['current_price'] = price
            pos['value'] = pos['shares'] * price
    
    # Calculate total portfolio value
    total_value = self.cash + sum(pos['value'] for pos in self.positions.values())
    self.portfolio_value = total_value
    
    # Update weights
    for ticker in self.positions:  # 🔴 ANOTHER LOOP
        if total_value > 0:
            self.positions[ticker]['weight'] = self.positions[ticker]['value'] / total_value
```

**Issues:**
- Nested dictionary operations (slow)
- Loop per ticker, called 2-3 times per day
- Recalculates total_value from scratch each time

**Suggested Fix:**
```python
# Store positions as DataFrame instead of dict
self.positions_df = pd.DataFrame(columns=['ticker', 'shares', 'entry_price'])

def update_positions_vectorized(self, prices_series):
    """
    Update all positions at once.
    
    Args:
        prices_series: pd.Series with ticker index
    """
    # Update prices (vectorized)
    self.positions_df['current_price'] = prices_series
    
    # Calculate values (vectorized)
    self.positions_df['value'] = (
        self.positions_df['shares'] * self.positions_df['current_price']
    )
    
    # Calculate total (single operation)
    total_value = self.cash + self.positions_df['value'].sum()
    
    # Calculate weights (vectorized)
    self.positions_df['weight'] = self.positions_df['value'] / total_value if total_value > 0 else 0
    
    return total_value
```

**Performance Gain:** ~10x faster per call

---

### 3. **Signal Change Detection in `update_signals()` (Lines 253-268)**

**Current Code:**
```python
def update_signals(self, signals: Dict[str, int], prices: Dict[str, float], date: pd.Timestamp):
    # Track which signals changed
    signal_changes = {}
    for ticker, new_signal in signals.items():  # 🔴 LOOP PER TICKER
        pos = self.positions.get(ticker, {})
        current_shares = pos.get('shares', 0)
        current_signal = np.sign(current_shares)
        
        if new_signal != current_signal:
            signal_changes[ticker] = {
                'old': current_signal,
                'new': new_signal,
                'price': prices[ticker]
            }
```

**Issues:**
- Manual loop to detect changes
- Multiple dictionary lookups per ticker per day

**Suggested Fix:**
```python
# Pre-compute signal changes for entire backtest period upfront
signal_changes = signals_df.diff()  # One operation for entire history
entry_mask = (signals_df != 0) & (signals_df.shift(1) == 0)  # New entries
exit_mask = (signals_df == 0) & (signals_df.shift(1) != 0)  # Exits
flip_mask = (signals_df != 0) & (signals_df.shift(1) != 0) & (signals_df != signals_df.shift(1))  # Flips

# Then apply trades only where masks are True
```

**Performance Gain:** ~100x faster (one-time calculation vs per-day loop)

---

### 4. **Rebalancing Check (Lines 111-153)**

**Current Code:**
```python
def check_rebalance_needed(self, signals: Dict[str, int]) -> bool:
    active_signals = self.get_active_signals(signals)
    n_active = len(active_signals)
    
    if n_active <= 1:
        return False
    
    target_weight = 1.0 / n_active
    
    # Check drift for active positions only
    for ticker in active_signals:  # 🔴 LOOP PER ACTIVE ASSET
        current_weight = self.positions.get(ticker, {}).get('weight', 0)
        
        # Calculate weight among active positions only
        total_active_value = sum(
            self.positions.get(t, {}).get('value', 0) 
            for t in active_signals  # 🔴 NESTED LOOP
        )
        
        if total_active_value > 0:
            weight_among_active = self.positions.get(ticker, {}).get('value', 0) / total_active_value
            drift = abs(weight_among_active - target_weight)
            
            if drift > self.config.rebalance_threshold:
                return True
    
    return False
```

**Issues:**
- Nested loops (active signals × active signals)
- Recalculates `total_active_value` for each ticker (redundant)
- Called every single day

**Suggested Fix:**
```python
# Pre-determine rebalancing dates based on frequency
if config.rebalance_frequency == 'monthly':
    rebal_dates = pd.date_range(start, end, freq='M')
elif config.rebalance_frequency == 'quarterly':
    rebal_dates = pd.date_range(start, end, freq='Q')
else:  # drift-based
    # Vectorized drift calculation
    weights = position_values.div(position_values.sum(axis=1), axis=0)
    target_weights = 1.0 / (signals_df != 0).sum(axis=1)
    drift = (weights - target_weights).abs()
    needs_rebalance = drift > threshold
    rebal_dates = needs_rebalance[needs_rebalance.any(axis=1)].index
```

**Performance Gain:** ~50x faster (pre-computed vs daily check)

---

## 🟡 MODERATE BOTTLENECKS

### 5. **Dictionary-Based State Storage**

**Current Approach:**
```python
self.positions = {}  # {ticker: {'shares': 0, 'value': 0, 'weight': 0}}
self.trades = []  # List of trade dicts
self.equity_curve = []  # List of state dicts
```

**Issues:**
- Dictionary operations slower than DataFrame operations
- List appends require reallocation
- Not vectorization-friendly

**Suggested Fix:**
```python
# Use DataFrames from the start
self.positions_df = pd.DataFrame(
    columns=['ticker', 'shares', 'entry_price', 'current_price', 'value', 'weight']
).set_index('ticker')

self.trades_df = pd.DataFrame(
    columns=['date', 'ticker', 'type', 'shares', 'price', 'value', 'tc']
)

self.equity_curve_series = pd.Series(dtype=float)  # Or pre-allocate array
```

**Performance Gain:** ~3-5x faster for large portfolios

---

### 6. **Multiple Passes Through Data**

**Current Flow:**
```
For each date:
    1. update_positions(prices)          # First pass
    2. update_signals(signals, prices)   # Second pass
    3. update_positions(prices)          # Third pass (unnecessary!)
    4. check_rebalance_needed()          # Fourth pass
    5. rebalance() if needed             # Fifth pass
    6. update_positions(prices)          # Sixth pass (unnecessary!)
    7. append to equity_curve            # Seventh operation
```

**Suggested Fix:**
```
# Single vectorized pass:
1. Pre-compute all signal changes (one operation)
2. Calculate position values for all dates (one operation)
3. Apply rebalancing at specific dates (small subset)
4. Calculate equity curve (one cumulative operation)
```

**Performance Gain:** ~6x faster (1 pass vs 6-7 passes)

---

## 🟢 MINOR OPTIMIZATIONS

### 7. **String Operations in Trades Recording**

```python
self.trades.append({
    'Date': date,
    'Ticker': ticker,
    'Type': 'Rebalance',  # String comparison slower than int
    ...
})
```

**Suggested Fix:**
Use enums or integer codes:
```python
class TradeType(IntEnum):
    ENTRY = 1
    EXIT = 2
    REBALANCE = 3
    FLIP = 4

self.trades.append({
    'type': TradeType.REBALANCE,  # Integer comparison faster
    ...
})
```

---

## 📊 EXPECTED PERFORMANCE IMPROVEMENTS

| Component | Current | Optimized | Speedup |
|-----------|---------|-----------|---------|
| Main loop | ~5-10s | ~0.1-0.2s | **50-100x** |
| update_positions | ~1-2s | ~0.05s | **20-40x** |
| Signal changes | ~0.5-1s | ~0.01s | **50-100x** |
| Rebalancing check | ~0.5s | ~0.01s | **50x** |
| **TOTAL** | **~7-13s** | **~0.2-0.5s** | **~20-50x** |

**Target:** <5 seconds for 10-year, 3-asset backtest ✅  
**Expected:** ~0.5 seconds (10x better than target!)

---

## 🎯 RECOMMENDED IMPLEMENTATION ORDER

### Phase 1: Quick Wins (1-2 hours)
1. **Convert state storage to DataFrames** (positions, trades, equity_curve)
2. **Remove redundant `update_positions()` calls** (only call once per date)
3. **Pre-allocate equity_curve** instead of appending

**Expected Gain:** ~5-10x faster

### Phase 2: Vectorize Core Loop (2-3 hours)
4. **Vectorize signal change detection** (calculate all at once)
5. **Vectorize position calculations** (all dates, all tickers)
6. **Pre-determine rebalancing dates** (avoid daily check)

**Expected Gain:** ~20-30x faster (cumulative)

### Phase 3: Full Refactor (3-4 hours if needed)
7. **Rewrite `run_multi_asset_backtest()` to be fully vectorized**
8. **Eliminate main date loop entirely**
9. **Use numpy arrays for hot paths**

**Expected Gain:** ~50-100x faster (cumulative)

---

## 🔧 IMPLEMENTATION STRATEGY

### Option A: Iterative Optimization (Recommended)
- Start with Phase 1 (quick wins)
- Measure performance after each change
- Stop when target (<5s) is achieved
- Keep code readable and maintainable

### Option B: Full Rewrite
- Rewrite entire portfolio_manager.py from scratch
- Use vectorized approach throughout
- Potentially more complex code
- Maximum performance gain

**Recommendation:** Start with **Option A**. If Phase 1 + Phase 2 get us to <1 second (likely), no need for full rewrite.

---

## 📝 CODE ARCHITECTURE SUGGESTIONS

### Current Architecture (Object-Oriented, Stateful):
```python
pm = PortfolioManager(config)
for date in dates:
    pm.update_positions(prices)
    pm.update_signals(signals, prices, date)
    pm.rebalance(...)
```

### Vectorized Architecture (Functional, Stateless):
```python
def calculate_portfolio_equity(signals_df, prices_df, config):
    """
    Pure function: given inputs, calculate outputs.
    No state, no loops, fully vectorized.
    """
    # All calculations happen on entire DataFrames
    positions = calculate_positions_vectorized(signals_df, prices_df, config)
    trades = detect_trades_vectorized(positions)
    equity = calculate_equity_vectorized(positions, trades, config)
    return equity, trades
```

**Benefits:**
- Easier to test (no state)
- Easier to parallelize (no side effects)
- Much faster (vectorized operations)
- More functional programming style

**Drawbacks:**
- Different mental model
- Harder to debug step-by-step
- Less intuitive for sequential logic

---

## 🧪 TESTING STRATEGY

After each optimization:

1. **Correctness Test:**
   ```python
   # Run old vs new, compare results
   equity_old, trades_old = run_old_version(...)
   equity_new, trades_new = run_new_version(...)
   
   assert np.allclose(equity_old, equity_new, rtol=0.01)  # Within 1%
   assert len(trades_old) == len(trades_new)
   ```

2. **Performance Test:**
   ```python
   import time
   
   start = time.time()
   result = run_backtest(...)
   elapsed = time.time() - start
   
   print(f"Execution time: {elapsed:.2f}s")
   assert elapsed < 5.0  # Must be under 5 seconds
   ```

3. **Regression Test:**
   ```python
   # Save baseline results
   baseline = run_backtest(test_data)
   baseline.to_pickle('tests/baseline_results.pkl')
   
   # Compare future runs
   current = run_backtest(test_data)
   assert current.equals(baseline)
   ```

---

## 💡 ADDITIONAL NOTES

### Why Vectorization Matters:
- **NumPy/Pandas are written in C** - 50-100x faster than Python loops
- **SIMD operations** - process multiple values simultaneously
- **Memory locality** - better cache utilization
- **Less Python overhead** - fewer function calls, less bytecode execution

### When NOT to Vectorize:
- Complex conditional logic that's hard to express vectorized
- Code that's already fast enough
- When it significantly hurts readability

### Tools for Profiling:
```python
# Use cProfile
python -m cProfile -o profile.stats scripts/test_multi_strategy.py

# Analyze results
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)

# Or use line_profiler for line-by-line analysis
@profile
def run_backtest():
    ...
    
kernprof -l -v script.py
```

---

**Next Steps:**
1. Review this analysis
2. Decide on implementation approach (Phase 1 only? Phase 1+2?)
3. Create backup branch: `git checkout -b optimization-backup`
4. Start implementing chosen optimizations
5. Test after each change
6. Commit when target achieved

