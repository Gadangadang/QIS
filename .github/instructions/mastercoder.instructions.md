# Coding Guidelines for QuantTrading Project

## Core Philosophy
- **Production-ready code**: Write institutional-quality code from the start
- **Test-driven development**: Every feature gets comprehensive tests
- **Documentation**: Clear docstrings, type hints, and inline comments
- **Performance**: Vectorized operations over loops wherever possible

## Code Structure & Style

### Python Standards
- Follow PEP 8 style guide
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use dataclasses for data structures
- Prefer composition over inheritance

### Naming Conventions
```python
# Classes: PascalCase
class BacktestOrchestrator:

# Functions/methods: snake_case
def run_backtest():

# Constants: UPPER_SNAKE_CASE
MAX_POSITION_SIZE = 1.0

# Private methods: _leading_underscore
def _validate_params():
```

### Error Handling
- Raise informative exceptions with context
- Validate inputs early (fail fast)
- Use RuntimeError for state violations
- Use ValueError for invalid parameters

## Architecture Principles

### Separation of Concerns
- **Signals**: Pure signal generation logic (no portfolio management)
- **Portfolio**: Position sizing and execution (no signal logic)
- **Orchestrator**: High-level workflow coordination
- **Utils**: Reusable helper functions

### Vectorization
- Use pandas/numpy operations instead of loops
- Avoid `.apply()` when vectorized alternatives exist
- Process entire DataFrames at once
- Example: `df['signal'] = (df['close'] > df['sma']).astype(int)`

### Configuration Over Hardcoding
- Use config dictionaries for parameters
- Make everything configurable via constructor args
- Default values should be sensible
- Document parameter meanings in docstrings

## Testing Requirements

### Test Coverage
- **Minimum 80% coverage** for new code
- Every new feature requires tests
- Test both happy path and edge cases
- Use pytest fixtures for setup

### Test Structure
```python
def test_feature_name():
    # Arrange: Set up test data
    data = create_test_data()
    
    # Act: Execute the feature
    result = feature(data)
    
    # Assert: Verify expectations
    assert result.metric > expected_value
```

### Test Naming
- `test_<feature>_<scenario>_<expected_outcome>`
- Example: `test_optimizer_invalid_params_raises_error`

## Performance Guidelines

### Pandas Best Practices
- Use `.loc[]` and `.iloc[]` for indexing (not chained indexing)
- Avoid loops: use `.groupby()`, `.rolling()`, `.expanding()`
- Use `.copy()` explicitly to avoid SettingWithCopyWarning
- Set index for time series data

### Memory Management
- Don't copy large DataFrames unnecessarily
- Use views when possible
- Clean up in finally blocks or context managers
- Use generators for large iterations

### Computation
- Leverage NumPy broadcasting
- Use built-in pandas functions (they're optimized)
- Profile before optimizing (measure, don't guess)
- Cache expensive calculations when appropriate

## Documentation Standards

### Docstrings
Use Google-style docstrings:
```python
def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Series of period returns (e.g., daily)
        risk_free_rate: Annual risk-free rate (default: 0.0)
    
    Returns:
        Annualized Sharpe ratio
        
    Raises:
        ValueError: If returns series is empty
        
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015])
        >>> sharpe = calculate_sharpe(returns)
        >>> print(f"Sharpe: {sharpe:.2f}")
    """
```

### Comments
- Explain **why**, not **what** (code shows what)
- Comment complex algorithms or non-obvious logic
- Use TODO/FIXME/NOTE markers for future work
- Keep comments up-to-date with code changes

## Workflow Practices

### Git Commits
- **Conventional commits**: `feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `chore:`
- Small, atomic commits (one logical change per commit)
- Descriptive commit messages explaining context
- Reference issue numbers when applicable

### Before Pushing
1. Run all tests: `pytest tests/ -v`
2. Check test coverage: `pytest tests/ --cov`
3. Verify no obvious errors
4. Ensure CI will pass

### Code Review Checklist
- [ ] Type hints present
- [ ] Docstrings complete
- [ ] Tests added/updated
- [ ] No hardcoded values
- [ ] Error handling appropriate
- [ ] Vectorized where possible
- [ ] No copy-paste duplication

## Project-Specific Conventions

### Signal Generators
- Inherit from `BaseSignal`
- Implement `generate()` method
- Return DataFrame with 'signal' column (-1, 0, 1)
- Never modify input DataFrame (use .copy())
- Validate parameters in `__init__`

### Backtesting
- Use `BacktestOrchestrator` for multi-strategy workflows
- Use `PortfolioManagerV2` for single-strategy tests
- Always include OOS validation
- Export results to `results/` directory

### Configuration
- Portfolio-level config in orchestrator
- Strategy-level config per signal
- Use CONFIG dictionaries in notebooks
- Never hardcode paths or magic numbers

## Anti-Patterns to Avoid

### Don't Do This
```python
# ❌ Loop through DataFrame
for i in range(len(df)):
    df.loc[i, 'signal'] = calculate(df.loc[i, 'price'])

# ❌ Chained indexing
df['price'][df['date'] > '2020-01-01'] = 100

# ❌ Hardcoded values
if returns > 0.15:  # What is 0.15?

# ❌ Silent failures
try:
    result = risky_operation()
except:
    pass
```

### Do This Instead
```python
# ✅ Vectorized operation
df['signal'] = df['price'].apply(calculate)

# ✅ Proper indexing
df.loc[df['date'] > '2020-01-01', 'price'] = 100

# ✅ Named constants
RETURN_THRESHOLD = 0.15  # 15% annual return target
if returns > RETURN_THRESHOLD:

# ✅ Explicit error handling
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise
```

## When Adding New Features

### Planning Phase
1. Check if similar functionality exists
2. Determine where it fits in architecture
3. Plan the API (inputs/outputs)
4. Consider edge cases

### Implementation Phase
1. Write the interface/signature first
2. Add docstring with examples
3. Implement the logic (vectorized if possible)
4. Add input validation
5. Handle errors gracefully

### Testing Phase
1. Write tests for normal usage
2. Test edge cases (empty data, invalid params)
3. Test error conditions
4. Verify performance on realistic data
5. Check test coverage percentage

### Integration Phase
1. Update orchestrator if needed
2. Add to template notebooks if relevant
3. Update README if user-facing
4. Commit with descriptive message

## Performance Targets

### Backtesting Speed
- Single strategy, 10 years daily data: < 1 second
- Multi-strategy (5 strategies): < 5 seconds
- Walk-forward optimization (4 params): < 2 minutes

### Test Suite
- All 140+ tests should pass
- Total runtime: < 5 seconds
- No flaky tests allowed

### Memory
- Can backtest 10 strategies on 10 assets without issues
- No memory leaks in long-running processes
- Clean up resources in notebooks

## Questions to Ask Before Coding

1. **Is this vectorizable?** (If yes, vectorize it)
2. **What could go wrong?** (Add validation/error handling)
3. **How will this be tested?** (Write tests alongside code)
4. **Is this configurable?** (Make parameters explicit)
5. **Will this scale?** (Test with realistic data sizes)
6. **Is the API clear?** (Good names, type hints, docstring)
7. **Does it follow patterns?** (Consistent with existing code)
8. **Can this be refactored?** (Code should be modular and refactored when possible)

## Resources

- **Style Guide**: PEP 8
- **Type Hints**: PEP 484
- **Docstrings**: Google Style Guide
- **Testing**: pytest documentation
- **Pandas**: Pandas User Guide (vectorization section)

## Remember

> "Code is read more often than it is written."  
> Write code that your future self will thank you for.

> "Make it work, make it right, make it fast."  
> In that order. Don't optimize prematurely.

> "Tests are not optional."  
> They're your safety net and documentation.

---

**When implementing changes, always:**
1. Follow these guidelines consistently
2. Write tests for new functionality
3. Update documentation as needed
4. Run the test suite before committing
5. Confirm understanding by acknowledging key requirements