# GitHub Copilot Agent Instructions for QIS Project

> **For Copilot Agents (PR automation)** - Extends `.github/copilot-instructions.md` (chat Copilot)  
> **Purpose:** Rules for autonomous code generation via `@copilot` in PR conversations

---

## 🎯 Core Mission

Build **production-ready quantitative trading infrastructure** with:
- ✅ Institutional-quality code (hedge fund standards)
- ✅ Comprehensive test coverage (80% minimum, **exception: plotting/visualization code**)
- ✅ Performance-first (vectorized operations, < 1s for single strategy backtest)
- ✅ Type safety (full type hints, validated inputs)
- ✅ Clear documentation (Google-style docstrings with examples)

---

## 📋 Code Quality Standards (Non-Negotiable)

### **1. Type Hints & Validation**
```python
# ✅ ALWAYS DO THIS
def calculate_sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    annualization_factor: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate (default: 2%)
        annualization_factor: Trading days per year (default: 252)
    
    Returns:
        Annualized Sharpe ratio
        
    Raises:
        ValueError: If returns is empty or annualization_factor <= 0
        
    Example:
        >>> returns = pd.Series([0.01, -0.005, 0.02])
        >>> sharpe = calculate_sharpe(returns)
        >>> assert sharpe > 0
    """
    if len(returns) == 0:
        raise ValueError("Returns series cannot be empty")
    if annualization_factor <= 0:
        raise ValueError(f"annualization_factor must be positive, got {annualization_factor}")
    
    # Implementation...
```

### **2. Vectorization (Performance Critical)**
```python
# ❌ NEVER DO THIS - Looping through DataFrames
for i in range(len(df)):
    if df.loc[i, 'close'] > df.loc[i, 'sma']:
        df.loc[i, 'signal'] = 1

# ✅ ALWAYS DO THIS - Vectorized operations
df['signal'] = (df['close'] > df['sma']).astype(int)
```

**Vectorization Rules:**
- Use `.rolling()`, `.expanding()`, `.shift()` for time-series operations
- Avoid `.apply()` unless absolutely necessary (check for vectorized alternative first)
- Use boolean indexing: `df.loc[df['price'] > 100, 'signal'] = 1`
- Leverage NumPy broadcasting when possible

### **3. Immutability & Side Effects**
```python
# ❌ NEVER DO THIS - Modifying input
def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    df['signal'] = calculate_signal(df)  # Modifies caller's DataFrame!
    return df

# ✅ ALWAYS DO THIS - Copy first
def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Prevent side effects
    df['signal'] = calculate_signal(df)
    return df
```

### **4. Error Handling (Fail Fast)**
```python
# ✅ Validate inputs early
def __init__(self, lookback_period: int = 50):
    if lookback_period <= 0:
        raise ValueError(f"lookback_period must be positive, got {lookback_period}")
    if lookback_period > 500:
        raise ValueError(f"lookback_period too large (max 500), got {lookback_period}")
    self.lookback_period = lookback_period

# ✅ Informative exceptions
if capital_allocated > total_capital:
    raise RuntimeError(
        f"Allocated capital (${capital_allocated:,.0f}) exceeds "
        f"total capital (${total_capital:,.0f})"
    )

# ❌ NEVER use bare except
try:
    risky_operation()
except:  # DON'T DO THIS
    pass
```

---

## 🧪 Testing Requirements

### **Coverage Rules**
- **Core business logic**: 80% minimum coverage
- **Data processing**: 80% minimum coverage  
- **Signal generators**: 80% minimum coverage
- **Portfolio management**: 80% minimum coverage
- **Risk calculations**: 90% minimum coverage (critical for correctness)
- **Plotting/Visualization**: **NO UNIT TESTS REQUIRED** ⚠️
- **HTML/Report generation**: **NO UNIT TESTS REQUIRED** ⚠️

### **Plotting Exception Rule** ⚠️
**DO NOT write unit tests for:**
- Matplotlib plotting functions (`.plot()`, `.subplot()`, `.axhline()`, etc.)
- Plotly chart generation
- HTML report rendering
- Visualization formatting (colors, fonts, layout)

**Why:**
- Matplotlib mocking is brittle across Python versions
- Visual output requires manual inspection anyway
- Tests add complexity without value
- Focus testing on calculation correctness, not rendering

**Instead:**
- ✅ Test the **data preparation** for plots (e.g., `prepare_equity_curve_data()`)
- ✅ Test that plotting methods **execute without errors** (smoke tests)
- ✅ Validate plots **manually** during development

**Example:**
```python
# ✅ GOOD - Test data preparation
def test_prepare_monthly_returns_data():
    """Test monthly returns aggregation for heatmap."""
    daily_returns = pd.Series([0.01, 0.02, -0.01], 
                               index=pd.date_range('2023-01-01', periods=3))
    monthly = prepare_monthly_returns(daily_returns)
    assert len(monthly) == 1  # All in same month
    assert monthly.iloc[0] == pytest.approx(0.0197, rel=1e-3)

# ✅ ACCEPTABLE - Smoke test (optional)
def test_plot_equity_curve_executes():
    """Verify plot_equity_curve runs without crashing."""
    result = BacktestResult(equity_curve=sample_data, trades=sample_trades)
    result.plot_equity_curve()  # Should not raise

# ❌ BAD - Don't mock matplotlib internals
@patch('matplotlib.pyplot.subplots')
def test_plot_creates_two_subplots(mock_subplots):  # DON'T DO THIS
    # Complex mocking that breaks across Python versions...
```

### **Test Structure (AAA Pattern)**
```python
def test_sma_crossover_signal_generates_buy_on_uptrend():
    """Test SMA crossover generates buy signal when fast > slow."""
    # Arrange
    prices = pd.DataFrame({
        'close': [100, 102, 105, 108, 110],
        'date': pd.date_range('2023-01-01', periods=5)
    }).set_index('date')
    signal_gen = SMACrossover(fast_period=2, slow_period=3)
    
    # Act
    result = signal_gen.generate(prices)
    
    # Assert
    assert 'Signal' in result.columns
    assert result['Signal'].iloc[-1] == 1  # Buy signal
    assert result['Signal'].iloc[0] == 0   # Warm-up period
```

### **Test Naming Convention**
`test_<component>_<scenario>_<expected_outcome>`

Examples:
- `test_sharpe_calculation_with_positive_returns_returns_positive_value`
- `test_position_sizer_zero_volatility_raises_error`
- `test_backtest_orchestrator_invalid_capital_allocation_raises_runtime_error`

---

## 🏗️ Architecture Patterns

### **Separation of Concerns**
```python
# ✅ GOOD - Clear separation
class MomentumSignal(SignalModel):
    """Generates momentum signals (no portfolio logic)."""
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Pure signal logic only
        return df

class PortfolioManagerV2:
    """Manages portfolio (no signal logic)."""
    def execute_signals(self, signals: pd.DataFrame):
        # Position sizing, risk management, execution
        pass

# ❌ BAD - Mixed concerns
class MomentumStrategy:
    """Signals + portfolio management mixed."""
    def run(self, df: pd.DataFrame):
        signals = self._generate_signals(df)  # Mixing concerns
        positions = self._size_positions(signals)
        # Hard to test, hard to reuse
```

### **Configuration Over Hardcoding**
```python
# ❌ BAD - Hardcoded values
def calculate_position_size(price, volatility):
    max_risk = 10000  # What is this?
    return max_risk / volatility

# ✅ GOOD - Configurable
class VolatilityPositionSizer:
    def __init__(
        self,
        max_risk_per_trade: float = 10000.0,
        volatility_multiplier: float = 2.0
    ):
        """
        Args:
            max_risk_per_trade: Maximum $ risk per trade
            volatility_multiplier: ATR multiplier for stop distance
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.volatility_multiplier = volatility_multiplier
```

---

## 📊 Quantitative Finance Specific Rules

### **1. Financial Calculations (High Standards)**
- Use `Decimal` for currency calculations when precision matters
- Document all formulas with references (e.g., "Sharpe ratio from Sharpe (1994)")
- Validate against known benchmarks in tests
- Handle edge cases: zero returns, negative prices, missing data

### **2. Time Series Handling**
```python
# ✅ Always set datetime index for time series
df.set_index('date', inplace=True)

# ✅ Handle forward-looking bias
df['sma'] = df['close'].rolling(20).mean()
df['signal'] = (df['close'] > df['sma'].shift(1)).astype(int)  # Use yesterday's SMA

# ✅ Warm-up periods
df['signal'] = df['signal'].fillna(0)  # No signal during warm-up
```

### **3. Backtesting Integrity**
- No lookahead bias (use `.shift()` to reference previous periods)
- Account for transaction costs and slippage
- Use realistic order fills (no assuming exact limit fills)
- Validate strategy on out-of-sample data

### **4. Risk Metrics Consistency**
- **Sharpe Ratio**: Risk-free rate = 2% annual (0.02)
- **Annualization**: 252 trading days for daily data
- **Drawdown**: Peak-to-trough percentage decline
- **Win Rate**: Winning trades / total trades
- **Profit Factor**: Gross profit / gross loss

---

## 🚫 Common Anti-Patterns (Never Do These)

| ❌ Anti-Pattern | ✅ Correct Pattern |
|----------------|-------------------|
| `for i in range(len(df)): df.loc[i, 'x'] = ...` | `df['x'] = vectorized_operation()` |
| `df['col'] = df.apply(lambda x: func(x))` | `df['col'] = func(df['input_col'])` (vectorized) |
| Hardcoded `0.05` in calculations | `STOP_LOSS_PCT = 0.05` (named constant) |
| `try: ... except: pass` | `try: ... except SpecificError as e: logger.error(...)` |
| Modifying input DataFrames | `df = df.copy()` first |
| No type hints | Full type hints on all signatures |
| Missing docstrings | Google-style docstrings with examples |
| Tests with `test_1`, `test_2` | Descriptive test names |
| Classes in notebooks | Classes in `core/` modules, import in notebooks |
| Timestamp div IDs in HTML | Use UUIDs or static strings |

---

## 🔄 Development Workflow

### **When Adding New Features:**

1. **Plan the API**
   - Inputs, outputs, edge cases
   - Check for existing similar functionality

2. **Write signature + docstring first**
   ```python
   def new_feature(param: type) -> return_type:
       """
       Brief description.
       
       Args:
           param: Description
       
       Returns:
           Description
           
       Raises:
           Error: When this happens
           
       Example:
           >>> result = new_feature(value)
           >>> assert result > 0
       """
       pass  # TODO: Implement
   ```

3. **Implement with validation**
   - Validate inputs early
   - Use vectorized operations
   - Avoid side effects

4. **Write tests alongside**
   - Test happy path
   - Test edge cases
   - Test error conditions
   - **Skip tests for plotting/visualization**

5. **Verify before committing**
   ```bash
   pytest tests/ -v  # All tests must pass
   pytest tests/ --cov  # Check coverage
   ```

### **When Fixing Bugs:**

1. **Reproduce with a test first**
   ```python
   def test_bug_reproduction():
       # Minimal reproduction case
       result = buggy_function(problematic_input)
       assert result == expected  # Fails initially
   ```

2. **Fix the code**

3. **Verify test passes**

4. **Add regression test**

---

## ⚡ Performance Targets

| Operation | Target Time | Notes |
|-----------|-------------|-------|
| Single strategy backtest (10 years daily) | < 1 second | 2,520 data points |
| Multi-strategy backtest (5 strategies) | < 5 seconds | Vectorized execution |
| Full test suite | < 10 seconds | 800+ tests |
| Signal generation (1 year) | < 100ms | Vectorized operations |

**If performance lags:**
1. Profile with `cProfile` or `line_profiler`
2. Check for loops (should be rare)
3. Verify vectorization
4. Cache expensive calculations

---

## 📝 Commit Conventions

Use conventional commits:
- `feat: add RSI signal generator with configurable periods`
- `fix: correct Sharpe ratio annualization factor`
- `test: add edge case tests for position sizer`
- `refactor: extract risk management into separate class`
- `docs: update signal generator usage examples`
- `perf: vectorize portfolio rebalancing logic`

---

## 🎯 Agent-Specific Guidelines

### **When Generating Code:**
1. ✅ Always include type hints
2. ✅ Always include Google-style docstrings with examples
3. ✅ Always validate inputs (fail fast)
4. ✅ Always use vectorized operations for pandas/numpy
5. ✅ Always write tests (except for plotting/visualization)
6. ✅ Always handle edge cases (empty data, missing values, etc.)
7. ⚠️ **NEVER** write unit tests for matplotlib/plotly plotting code
8. ⚠️ **NEVER** mock matplotlib internals (breaks across Python versions)

### **When Asked to Increase Test Coverage:**
1. ✅ Focus on business logic (calculations, data processing)
2. ✅ Test edge cases and error handling
3. ✅ Test data transformations
4. ⚠️ **SKIP** plotting/visualization functions
5. ⚠️ **SKIP** HTML report generation
6. ✅ Test data preparation FOR plots (not the plotting itself)

### **When Tests Fail:**
1. Analyze the error message
2. Check if it's a mocking issue (especially matplotlib)
3. If matplotlib mocking is complex → suggest removing the test
4. If business logic issue → fix the implementation
5. Never create overly complex mocks just to pass tests

### **Decision Framework for Testing:**
```
Is it a calculation/business logic function?
  └─> YES → Write comprehensive unit tests (80% coverage)
  └─> NO ↓

Is it a data transformation/processing function?
  └─> YES → Write comprehensive unit tests (80% coverage)
  └─> NO ↓

Is it a plotting/visualization function?
  └─> YES → NO UNIT TESTS (manual validation only)
  └─> NO ↓

Is it an HTML/report generation function?
  └─> YES → Smoke test only (verify executes without error)
  └─> NO ↓

Default → Write tests with 80% coverage goal
```

---

## ✅ Quality Checklist (Use Before Committing)

- [ ] Type hints on all function signatures
- [ ] Google-style docstrings with examples
- [ ] Input validation with informative errors
- [ ] Vectorized operations (no loops through DataFrames)
- [ ] Immutable operations (`.copy()` for DataFrames)
- [ ] Tests written (AAA pattern, descriptive names)
- [ ] **Tests skipped for plotting/visualization code**
- [ ] Test coverage ≥ 80% (excluding plots)
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No lint errors
- [ ] Conventional commit message

---

## ��� Learning Resources

When uncertain about best practices:
- **Vectorization**: Check pandas documentation for vectorized alternatives
- **Type hints**: Use `mypy` for type checking
- **Testing**: Follow AAA pattern (Arrange, Act, Assert)
- **Financial calcs**: Reference academic papers (Sharpe 1994, etc.)
- **Code quality**: Read `.clinerules` and `.github/copilot-instructions.md`

---

**Remember:** Code quality > Speed. Take time to do it right.
Production quant systems handle real money. 💰
