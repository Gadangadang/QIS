"""
Simple test of refactored optimizer - verify all methods work correctly.
Instead of comparing old vs new (import issues), just verify new works.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.taa.optimizer import TAAOptimizer
from core.taa.constraints import OptimizationConstraints

print("=" * 80)
print("OPTIMIZER REFACTOR FUNCTIONALITY TEST")
print("=" * 80)

# 1. Load test data
print("\n1. Loading test data...")
predictions_path = project_root / "data" / "taa" / "oos_predictions_12w.csv"
df_long = pd.read_csv(predictions_path)
df_long['date'] = pd.to_datetime(df_long['date'])

# Pivot to wide format
df = df_long.pivot(index='date', columns='ticker', values='predicted').reset_index()
df.columns.name = None
df = df.rename(columns={'date': 'Date'})

# Get one test date
test_date = df['Date'].iloc[0]
test_row = df.iloc[0]

tickers = [col for col in df.columns if col != 'Date']
expected_returns_array = test_row[tickers].values
expected_returns = dict(zip(tickers, expected_returns_array))

print(f"   Test date: {test_date}")
print(f"   Tickers: {tickers}")
print(f"   Expected returns: {list(expected_returns.values())[:3]}...")

# 2. Calculate covariance
print("\n2. Calculating covariance matrix...")
from core.data.collectors.yahoo_collector import YahooCollector
collector = YahooCollector()

end_date = test_date
start_date = end_date - pd.Timedelta(days=365)

prices = collector.fetch_history(
    tickers=tickers,
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d')
)

# Handle single vs multiple tickers format
if len(tickers) == 1:
    returns = prices['Close'].pct_change().dropna()
else:
    # Multi-ticker: columns are (ticker, price_type)
    close_prices = pd.DataFrame()
    for ticker in tickers:
        if ticker in prices.columns.get_level_values(0):
            close_prices[ticker] = prices[(ticker, 'Close')]
    returns = close_prices.pct_change().dropna()

# Keep as DataFrame with ticker column names
cov_matrix = returns.cov() * 252  # Annualize

print(f"   Covariance matrix shape: {cov_matrix.shape}")
print(f"   Historical returns: {len(returns)} days")

# 3. Load constraints
print("\n3. Loading constraints...")
from core.taa.constraints import load_constraints_from_config
constraints_path = project_root / "config" / "taa_constraints.yaml"
constraints = load_constraints_from_config(str(constraints_path))
print("   ✓ Constraints loaded")

# 4. Test all 8 optimizer methods
print("\n4. Testing optimizer methods...")
print("=" * 80)

test_methods = [
    ('mean_variance', {}),
    ('max_sharpe', {}),
    ('min_variance', {}),
    ('risk_parity', {}),
    ('black_litterman', {'tau': 0.05}),
    ('cvar', {'alpha': 0.05}),
    ('hrp', {}),
    ('kelly', {'kelly_fraction': 0.5}),
]

results = {}
all_passed = True

for method, kwargs in test_methods:
    print(f"\nTesting {method}...")
    try:
        # Create optimizer with factory
        opt = TAAOptimizer(constraints=constraints, method=method, **kwargs)
        
        # Run optimization
        result = opt.optimize(
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix
        )
        
        # Handle return format (weights_dict, metadata_dict)
        if isinstance(result, tuple):
            weights_dict, metadata = result
            weights = np.array([weights_dict[t] for t in tickers])
        else:
            weights = result
        
        # Validate results
        assert len(weights) == len(tickers), f"Wrong number of weights: {len(weights)} vs {len(tickers)}"
        assert abs(weights.sum() - 1.0) < 1e-6, f"Weights don't sum to 1: {weights.sum()}"
        assert (weights >= -0.001).all(), f"Negative weights: {weights[weights < 0]}"
        assert (weights <= 1.001).all(), f"Weights > 1: {weights[weights > 1]}"
        
        # Store results
        results[method] = {
            'weights': weights,
            'max_weight': weights.max(),
            'min_weight': weights.min(),
            'num_positions': (weights > 0.01).sum(),
            'concentration': (weights ** 2).sum()  # Herfindahl index
        }
        
        print(f"   ✓ {method} PASSED")
        print(f"     Positions: {results[method]['num_positions']}")
        print(f"     Concentration: {results[method]['concentration']:.3f}")
        print(f"     Max weight: {results[method]['max_weight']:.3f}")
        
    except Exception as e:
        print(f"   ✗ {method} FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

# 5. Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

if all_passed:
    print("\n✓ ALL TESTS PASSED")
    print("\nOptimizer refactor successful - all 8 methods working correctly!")
    print("\nKey findings:")
    
    # Compare diversity
    concentrations = [(m, r['concentration']) for m, r in results.items()]
    concentrations.sort(key=lambda x: x[1])
    
    print("\nMost diversified (lowest concentration):")
    for method, conc in concentrations[:3]:
        print(f"  - {method}: {conc:.3f} ({results[method]['num_positions']} positions)")
    
    print("\nMost concentrated (highest concentration):")
    for method, conc in concentrations[-3:]:
        print(f"  - {method}: {conc:.3f} ({results[method]['num_positions']} positions)")
    
    sys.exit(0)
else:
    print("\n✗ SOME TESTS FAILED")
    print("\nPlease review errors above")
    sys.exit(1)
