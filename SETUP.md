# Development Setup Guide

Complete guide for setting up the QuantTrading development environment.

## Prerequisites

- Python 3.9 or higher
- Git
- (Optional) Conda for environment management

## Initial Setup

### 1. Clone Repository

```bash
git clone https://github.com/Gadangadang/QuantTrading.git
cd QuantTrading
```

### 2. Create Virtual Environment

**Option A: Using pip + venv**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

**Option B: Using Conda (Recommended)**
```bash
# From environment.yml
conda env create -f environment.yml
conda activate quant_trading

# Or create fresh environment
conda create -n quant_trading python=3.11
conda activate quant_trading
pip install -r requirements-dev.txt
```

### 3. Verify Installation

```bash
# Run test suite
pytest tests/ -v

# Check coverage
pytest tests/ --cov=core/portfolio --cov-report=html

# Verify all imports work
python -c "import pandas, numpy, matplotlib, plotly, scipy; print('✅ Success')"
```

## Development Workflow

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_portfolio_core.py -v

# With coverage
pytest tests/ --cov=core/portfolio --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=core/portfolio --cov-report=html
# Then open: htmlcov/index.html
```

### Code Quality

```bash
# Format code with black
black core/ signals/ tests/

# Check imports with isort
isort core/ signals/ tests/

# Lint with flake8
flake8 core/ signals/ tests/ --max-line-length=100 --ignore=E203,W503

# Run all quality checks
black core/ signals/ tests/ && isort core/ signals/ tests/ && flake8 core/ signals/ tests/
```

### Running Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

### Pre-Commit Checks

Before committing:

1. **Format code**: `black .`
2. **Run tests**: `pytest tests/ -v`
3. **Check coverage**: `pytest tests/ --cov=core/portfolio`
4. **Verify no lint errors**: `flake8 core/ signals/ tests/`

## CI/CD Pipeline

### GitHub Actions

**Location**: `.github/workflows/test.yml`

**Triggers**:
- Push to `main` or `develop` branches
- Pull requests to `main`

**Jobs**:
1. **Test** - Runs pytest across Python 3.9, 3.10, 3.11
2. **Lint** - Checks code quality with black, flake8, isort
3. **Coverage** - Uploads coverage to Codecov

**View Results**: Check the "Actions" tab on GitHub

### Dependabot

**Location**: `.github/dependabot.yml`

**Schedule**:
- Python dependencies: Weekly (Mondays)
- GitHub Actions: Monthly

**What It Does**:
- Automatically creates PRs for dependency updates
- Runs tests to verify updates don't break anything
- Labels PRs with `dependencies` tag

## Project Structure

```
QuantTrading/
├── .github/                    # GitHub configuration
│   ├── workflows/              # CI/CD pipelines
│   │   └── test.yml           # Test automation
│   └── dependabot.yml         # Dependency updates
│
├── core/                       # Core trading engine
│   └── portfolio/             # Portfolio management (v2)
│
├── signals/                    # Trading strategies
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py            # Shared fixtures
│   ├── test_portfolio_core.py # Portfolio tests
│   ├── README.md              # Test documentation
│   └── TEST_SUMMARY.md        # Coverage report
│
├── notebooks/                  # Research notebooks
│
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── environment.yml            # Conda environment
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation
```

## Common Issues

### Import Errors

If you get `ModuleNotFoundError`:
```bash
# Make sure you're in the project root
cd /path/to/QuantTrading

# Verify environment is activated
which python  # Should point to venv/conda environment

# Reinstall dependencies
pip install -r requirements-dev.txt
```

### Test Failures

If tests fail:
```bash
# Run with verbose output
pytest tests/ -vv

# Run single test for debugging
pytest tests/test_portfolio_core.py::TestFixedFractionalSizer::test_basic_position_size -vv

# Check if dependencies are correct versions
pip list | grep -E "(pytest|pandas|numpy)"
```

### Coverage Plugin Missing

If `--cov` flag not recognized:
```bash
pip install pytest-cov
```

## Tips for Development

1. **Use virtual environment**: Always activate before working
2. **Run tests frequently**: Quick feedback loop
3. **Check coverage**: Aim for >50% (currently 51%)
4. **Format before commit**: `black .` keeps code consistent
5. **Write tests first**: TDD approach for new features
6. **Document changes**: Update README/docstrings

## Resources

- **pytest docs**: https://docs.pytest.org/
- **black formatter**: https://black.readthedocs.io/
- **flake8 linter**: https://flake8.pycqa.org/
- **GitHub Actions**: https://docs.github.com/en/actions

## Getting Help

1. Check test output: `pytest tests/ -vv`
2. Read error messages carefully
3. Check `tests/TEST_SUMMARY.md` for coverage details
4. Review existing test examples in `tests/test_portfolio_core.py`
