# GitHub Copilot Instructions for QuantTrading

> **Companion to:** `.github/instructions/mastercoder.instructions.md` (code quality & style guide)  
> **This file:** Architecture, workflows, codebase-specific patterns

## Core Philosophy (Always Follow)
- Production-ready, institutional-quality code
- Test-driven development (80% minimum coverage)
- Vectorized pandas/numpy operations (never loop)
- Type hints + Google-style docstrings
- Configuration over hardcoding

## Architecture Overview

**Core Components:**
- `BacktestOrchestrator` - High-level multi-strategy workflow coordinator (method chaining pattern)
- `PortfolioManagerV2` - Single-strategy execution engine with modular risk/execution layers
- `SignalModel` (base) - Abstract signal generator (returns df with 'Signal' column: 1/0/-1)
- `RiskManager` - Position sizing, stop-loss, drawdown monitoring (separate from execution)
- `ExecutionEngine` - Transaction costs, slippage modeling
- `Portfolio` - Position and cash state management

**Data Flow:**
```
load_data() -> add_strategy() -> generate_signals() -> run_backtests() -> generate_comprehensive_reports()
```

**Key Pattern:** Orchestrator uses PortfolioManagerV2 internally. For single-strategy work, use PortfolioManagerV2 directly. For multi-strategy portfolios, always use BacktestOrchestrator.

**Separation of Concerns (Critical):**
- Signals: Pure signal logic, no portfolio management
- Portfolio: Position sizing/execution, no signal logic
- Orchestrator: High-level coordination only
- Utils: Reusable helpers

**V2 Architecture (Clean Separation):**
```
PortfolioManagerV2
├── Portfolio (state: positions, cash, value)
├── RiskManager (rules: stops, sizing, drawdown)
├── ExecutionEngine (costs: slippage, commissions)
└── BacktestResult (metrics: Sharpe, CAGR, trades)
```

## Critical Workflows

**Environment Setup:**
```bash
conda activate quant-paper-trading  # Always activate first
pytest tests/ -v  # Run all tests (must pass before committing)
pytest tests/test_signals.py -v  # Test specific module
```

**Signal Generator Development (Step-by-Step):**
1. Inherit from `SignalModel` in `signals/base.py`
2. Add type hints to `__init__` and validate parameters (fail fast)
3. Implement `generate(df: pd.DataFrame) -> pd.DataFrame`:
   - **Must** return `df.copy()` with 'Signal' column (-1/0/1)
   - **Never** modify input df (avoid side effects)
   - Set warm-up/burn-in period signals to 0
   - Use vectorized operations (`.rolling()`, `.shift()`, no loops)
4. Add Google-style docstring with Args, Returns, Example
5. Write tests in `tests/test_signals.py`:
   - Use fixtures: `uptrend_data`, `downtrend_data`, `sideways_data`
   - Test happy path, edge cases (empty df, invalid params)
   - Verify signal column exists and has correct values
   - Check no NaN in warm-up period after replacing with 0

**Signal Generator Template:**
```python
class MySignal(SignalModel):
    """
    Brief description.
    
    Args:
        param1: Description
        param2: Description
    
    Example:
        >>> signal = MySignal(param1=50)
        >>> df_with_signals = signal.generate(prices_df)
    """
    def __init__(self, param1: int = 50):
        if param1 <= 0:
            raise ValueError(f"param1 must be positive, got {param1}")
        self.param1 = param1
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()  # Critical: never modify input
        
**Vectorization (Performance Critical - See mastercoder.instructions.md):**
- **Never** loop through DataFrames with `.loc[i]` or `for i in range(len(df))`
- Use `.rolling()`, `.expanding()`, `.shift()` for time-series ops
- Avoid `.apply()` when vectorized alternatives exist
- Use `.loc[]` for indexing, never chained indexing
- Always use `.copy()` explicitly to avoid SettingWithCopyWarning
- Example: `df['signal'] = (df['close'] > df['sma']).astype(int)` not `.apply(lambda x: ...)`

**Performance Targets:**
- Single strategy, 10 years daily data: < 1 second
- Multi-strategy (5 strategies): < 5 seconds
- Test suite: < 5 seconds for all tests
        
        return df
```

**Backtest Pattern:**
```python
orchestrator = BacktestOrchestrator(config={'total_capital': 100000})
orchestrator.load_data(tickers=['ES', 'GC']) \
    .load_benchmark('SPY') \
    .add_strategy(name='Strategy1', signal_generator=MySignal(), 
                  assets=['ES'], capital_pct=0.6) \
    .generate_signals() \
    .run_backtests() \
    .generate_comprehensive_reports()  # Creates 3 HTML files
```

## Project-Specific Conventions

**Capital Allocation (Critical - causes RuntimeError if violated):**
- Use `capital_pct` (0.0-1.0) with `total_capital` in config OR
- Use `capital` (absolute $) per strategy
- **Never mix both** approaches in same orchestrator (raises ValueError)
- Orchestrator tracks allocated capital: `capital_pct` sum cannot exceed 1.0

**Risk Logging:**
- PortfolioManagerV2 accepts `risk_log_path` parameter
- Logs saved to `logs/risk_rejections_{strategy_name}.csv`
- Check logs after backtests to see rejected trades (position size limits, drawdown stops)

**Signal Column Names:**
- Primary signal: Must be named 'Signal' (capital S)
- Supporting columns: descriptive names (e.g., 'SMA_50', 'Regime', 'Z_Score')
- Signals must be discrete: 1 (long), 0 (flat), -1 (short)

**Vectorization (Performance Critical):**
- Never loop through DataFrames with `.loc[i]`
- Use `.rolling()`, `.expanding()`, `.shift()` for time-series ops
- Avoid `.apply()` when vectorized alternatives exist
- Example: `df['signal'] = (df['close'] > df['sma']).astype(int)` not `.apply(lambda x: ...)`
- Always use `.copy()` to avoid SettingWithCopyWarning

**Error Handling (Fail Fast):**
- `ValueError` for invalid parameters (capital_pct > 1.0, missing required args)
- `RuntimeError` for state violations (calling run_backtests before generate_signals)
- Always validate inputs early in `__init__` or method entry
- Raise informative exceptions with context: `ValueError(f"Expected X, got {actual}")`
- Never use bare `except:` - catch specific exceptions
- Log errors before re-raising: `logger.error(f"Operation failed: {e}")`

**HTML Reports:**
- `orchestrator.generate_comprehensive_reports()` creates 3 files:
  - Multi-Strategy Report (combined portfolio + benchmark)
  - Risk Dashboard (rejections, drawdowns, position sizes)
  - Individual Strategy Reports (per-strategy deep dive)
- Files saved to `results/html/` by default
- All use Plotly for interactive charts (equity curves, rolling metrics, trade analysis)

**Sharpe Ratio (Consistent Across All Modules):**
- Risk-free rate: 2% annual (0.02)
- Formula: `excess_returns = returns - (0.02/252); sharpe = sqrt(252) * excess_returns.mean() / returns.std()`
- Implementation in `core/portfolio/backtest_result.py:_calculate_sharpe()`

## Integration Points

**Asset Registry:**
- `core/asset_registry.py` is single source of truth for asset metadata
- Contains futures contract specs (ES: 50x multiplier, CL: 1000x, GC: 100x)
- Use `get_asset('ES')` to retrieve metadata
- Use `filter_by_class(AssetClass.COMMODITY_ENERGY)` for CL, NG
- Add new futures: create `AssetMetadata` with multiplier, tick_size, margin

**Position Sizers:**
- Base class: `PositionSizer` in `core/portfolio/position_sizers.py`
- Types: FixedFractionalSizer, ATRSizer, VolatilityScaledSizer, KellySizer, FuturesContractSizer
- Futures sizing: Use `FuturesContractSizer` for integer contract sizing
- Pass to PortfolioManagerV2: `position_sizer=FuturesContractSizer(contract_specs={'ES': 50})`
- All return integer share/contract counts

**Benchmark Analysis:**
- Load benchmark: `orchestrator.load_benchmark('SPY')`
- Reporters auto-calculate: alpha, beta, correlation, rolling correlation
## Common Pitfalls & Anti-Patterns

**Don't Do These:**
- ❌ Use timestamps in div IDs (causes mismatched Plotly render targets) - use UUIDs or static strings
- ❌ Use regular strings for JavaScript injection (must be f-strings for `.to_json()`)
- ❌ Forward-fill signals after exit conditions (causes positions to persist incorrectly)
- ❌ Create strategies without first calling `load_data()`
- ❌ Modify input DataFrames in signal generators (always `.copy()` first)
- ❌ Loop through DataFrames: `for i in range(len(df)): df.loc[i, 'col'] = ...`
- ❌ Hardcode magic numbers: `if returns > 0.15:` (use named constants)
- ❌ Silent failures: `try: risky_op() except: pass`
- ❌ Mix `capital` and `capital_pct` in same orchestrator

**Do These Instead:**
- ✅ Run `pytest tests/ -v` before every commit (CI requires 100% pass rate)
- ✅ Check `logs/risk_rejections_*.csv` after unexpected backtest results
- ✅ Use vectorized operations: `df['signal'] = (df['close'] > df['sma']).astype(int)`
- ✅ Named constants: `RETURN_THRESHOLD = 0.15; if returns > RETURN_THRESHOLD:`
- ✅ Explicit error handling with logging and context
- ✅ Write tests alongside code (AAA pattern: Arrange, Act, Assert)

## Development Workflow (Always Follow)

**Before Starting:**
1. Activate environment: `conda activate quant-paper-trading`
2. Check if similar functionality exists (avoid duplication)
3. Plan the API: inputs, outputs, edge cases

**While Coding:**
1. Write function signature with type hints first
2. Add Google-style docstring with example
3. Implement logic (vectorized, validated inputs)
4. Write tests alongside code
5. Run tests: `pytest tests/test_<module>.py -v`

**Before Committing:**
1. Run full test suite: `pytest tests/ -v`
2. Check coverage: `pytest tests/ --cov`
3. Verify no lint errors
4. Use conventional commits: `feat:`, `fix:`, `test:`, `refactor:`, `docs:`

## Quick Reference

**Test Naming:** `test_<feature>_<scenario>_<expected_outcome>`
**Commit Format:** `feat: add momentum signal with ATR filter`
**Docstring Style:** Google (Args, Returns, Raises, Example)
**Line Length:** 100 characters max
**Coverage Target:** 80% minimum for new code
- **Don't** modify input DataFrames in signal generators (always `.copy()` first)
- **Do** run `pytest` before every commit (CI requires 100% pass rate)
- **Do** check `logs/risk_rejections_*.csv` after unexpected backtest results
