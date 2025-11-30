# Unit Tests for QuantTrading

This directory contains unit and integration tests for the QuantTrading framework.

## Installation

Make sure pytest is installed:

```bash
conda activate quant_trading
pip install pytest pytest-cov
```

Or add to `environment.yml`:

```yaml
dependencies:
  - pytest>=7.0
  - pytest-cov
```

## Running Tests

**Run all tests:**
```bash
pytest tests/
```

**Run with verbose output:**
```bash
pytest tests/ -v
```

**Run specific test file:**
```bash
pytest tests/test_portfolio_core.py -v
```

**Run specific test class:**
```bash
pytest tests/test_portfolio_core.py::TestFixedFractionalSizer -v
```

**Run specific test:**
```bash
pytest tests/test_portfolio_core.py::TestFixedFractionalSizer::test_basic_position_size -v
```

**Run with coverage:**
```bash
pytest tests/ --cov=core --cov-report=html
```

Then open `htmlcov/index.html` to see coverage report.

**Run only unit tests:**
```bash
pytest tests/ -m unit
```

**Run integration tests:**
```bash
pytest tests/ -m integration
```

**Skip slow tests:**
```bash
pytest tests/ -m "not slow"
```

## Test Structure

### `test_portfolio_core.py`
Core portfolio management functionality:
- **Position Sizers**: FixedFractionalSizer, KellySizer, ATRSizer
- **RiskManager**: Stop-loss, take-profit, kill switches, heat limits
- **Portfolio**: Position tracking, P&L calculation
- **Integration**: Full backtest workflow

### `conftest.py`
Shared fixtures and configuration:
- Mock OHLCV data generators
- Mock signal generators
- Custom pytest markers

## Test Coverage

Current coverage focuses on:
- ✅ Position sizing calculations (fixed fractional, Kelly, ATR)
- ✅ Risk management (stops, limits, kill switches)
- ✅ Portfolio position tracking and P&L
- ✅ Edge cases (zero values, invalid inputs)

Future coverage:
- 🔲 Signal generation (momentum, mean reversion, trend following)
- 🔲 BacktestEngine validation
- 🔲 Reporter and dashboard generation
- 🔲 Strategy parameter optimization

## Writing New Tests

1. **Create test file**: `test_<module_name>.py`

2. **Import fixtures**:
```python
import pytest
from conftest import mock_ohlcv_data  # Shared fixtures auto-available
```

3. **Write test class**:
```python
class TestMyComponent:
    def test_basic_functionality(self):
        """Test description."""
        # Arrange
        component = MyComponent(param=value)
        
        # Act
        result = component.do_something()
        
        # Assert
        assert result == expected
```

4. **Use markers**:
```python
@pytest.mark.unit
def test_fast_unit_test():
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_full_backtest():
    pass
```

## Continuous Integration

Tests should be run before:
- Committing code changes
- Creating pull requests
- Deploying to production

Aim for:
- **Coverage**: >80% for core modules
- **Speed**: <10 seconds for unit tests
- **Isolation**: Each test independent, no side effects

## Troubleshooting

**Import errors:**
```python
# Make sure project root is in Python path
import sys
sys.path.append('..')
```

**Fixture not found:**
Check that `conftest.py` is in the tests/ directory.

**Test fails in CI but passes locally:**
Check for hardcoded paths, system-specific dependencies, or race conditions.

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [TDD in Python](https://testdriven.io/blog/modern-tdd/)
