# Parameter Optimization Guide

## Quick Start

The optimizer framework provides grid search and random search for strategy parameter tuning.

### Basic Usage

```python
from core.optimizer import ParameterOptimizer

# Define parameter grid
param_grid = {
    'lookback': [200, 250],
    'entry_threshold': [0.01, 0.02, 0.03],
    'stop_loss_pct': [0.08, 0.10, 0.12]
}

# Run grid search
optimizer = ParameterOptimizer(
    objective_fn=your_backtest_function,
    param_grid=param_grid,
    metric='sharpe',
    maximize=True
)

best_params, best_score, results = optimizer.grid_search()
```

### Test Script

Run the included test script:
```bash
python test_optimizer.py
```

This will:
- Test 4 parameter combinations
- Create individual folders for each trial in `logs/optimization/trials/`
- Generate HTML + PDF reports for each trial
- Save summary results in `logs/optimization/small_grid/`

### Trial Organization

Each trial gets its own folder:
```
logs/optimization/trials/
├── trial_001_lb200_sl8/
│   ├── report.html         # Interactive HTML report
│   ├── report.pdf          # PDF version (requires weasyprint/pdfkit)
│   ├── diagnostics.txt     # Diagnostic analysis
│   ├── stitched_equity.csv
│   ├── combined_returns.csv
│   └── trades_fold_*.csv
├── trial_002_lb200_sl10/
├── trial_003_lb250_sl8/
└── trial_004_lb250_sl10/
```

### PDF Generation

The optimizer automatically generates both HTML and PDF reports.

**Requirements:**
- HTML reports: Always generated (no dependencies)
- PDF reports: Requires one of:
  - `weasyprint` (recommended): `pip install weasyprint`
  - `pdfkit`: `pip install pdfkit` + `brew install wkhtmltopdf`

If neither is installed, PDF generation is skipped with a warning.

### Comparison Workflow

1. Run optimization: `python test_optimizer.py`
2. Compare trials by opening each `report.html` in browser
3. Check `logs/optimization/small_grid/optimization_results.csv` for rankings
4. Use best parameters from `logs/optimization/small_grid/best_params.json`

### Random Search

For larger parameter spaces, use random search:

```python
best_params, best_score, results = optimizer.random_search(
    n_iter=50,
    random_state=42
)
```

### Next Steps

- Integrate optimizer into walk-forward (per-fold optimization)
- Add parallel execution for faster optimization
- Add early stopping based on validation performance
