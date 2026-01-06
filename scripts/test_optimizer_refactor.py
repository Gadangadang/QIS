#!/usr/bin/env python3
"""
Test script to verify refactored optimizer.py produces same results as old version.

Compares outputs between:
- Old version: optimizer.py.old (individual optimizer classes)
- New version: optimizer.py (factory pattern with TAAOptimizer)
"""

import sys
sys.path.insert(0, '/Users/Sakarias/QuantTrading')

import pandas as pd
import numpy as np
from pathlib import Path

# Import new version
from core.taa.optimizer import TAAOptimizer
from core.taa.constraints import load_constraints_from_config

# Import old version by renaming the file temporarily
import importlib.util

def load_old_optimizer():
    """Load the old optimizer module by executing it."""
    import os
    old_path = "/Users/Sakarias/QuantTrading/core/taa/optimizer.py.old"
    
    if not os.path.exists(old_path):
        raise FileNotFoundError(f"File not found: {old_path}")
    
    # Create a namespace with required globals
    old_namespace = {
        '__name__': 'optimizer_old',
        '__file__': old_path,
        '__builtins__': __builtins__,
    }
    
    with open(old_path, 'r') as f:
        code = f.read()
    exec(code, old_namespace)
    
    # Create a simple object to hold the classes
    class OldOptModule:
        pass
    
    old_module = OldOptModule()
    for name in ['MeanVarianceOptimizer', 'MaxSharpeOptimizer', 'MinVarianceOptimizer',
                 'RiskParityOptimizer', 'BlackLittermanOptimizer', 'CVaROptimizer',
                 'HRPOptimizer', 'KellyOptimizer']:
        if name in old_namespace:
            setattr(old_module, name, old_namespace[name])
    
    return old_module

print("="*80)
print("OPTIMIZER REFACTOR VALIDATION TEST")
print("="*80)

# Load test data
print("\n1. Loading test data...")
data_dir = Path('/Users/Sakarias/QuantTrading/data/taa')
pred_4w_long = pd.read_csv(data_dir / 'oos_predictions_4w.csv')
predictions = pred_4w_long.pivot(index='date', columns='ticker', values='predicted')
predictions.index = pd.to_datetime(predictions.index)

# Get first prediction date
test_date = predictions.index[0]
expected_returns = predictions.loc[test_date].to_dict()
tickers = list(expected_returns.keys())

print(f"   Test date: {test_date}")
print(f"   Tickers: {tickers}")
print(f"   Expected returns: {list(expected_returns.values())[:3]}...")

# Load historical prices and calculate covariance
print("\n2. Calculating covariance matrix...")
from core.data.collectors.yahoo_collector import YahooCollector
collector = YahooCollector()

prices = collector.fetch_history(
    tickers=tickers,
    start_date='2010-01-01',
    end_date='2025-12-31'
)

close_prices = prices.xs('Close', level='Price', axis=1)
returns_df = close_prices.pct_change().dropna()

first_pred_date = predictions.index[0]
lookback_start = first_pred_date - pd.Timedelta(days=365)
historical_returns = returns_df.loc[lookback_start:first_pred_date]
cov_matrix = historical_returns.cov() * 252

print(f"   Covariance matrix shape: {cov_matrix.shape}")
print(f"   Historical returns: {len(historical_returns)} days")

# Load constraints
print("\n3. Loading constraints...")
constraints = load_constraints_from_config('/Users/Sakarias/QuantTrading/config/taa_constraints.yaml')
print(f"   ✓ Constraints loaded")

# Test each optimizer method
print("\n4. Testing optimizer methods...")
print("="*80)

# Load old optimizer module
old_opt = load_old_optimizer()

# Methods to test (mapping old class to new method name)
test_cases = [
    ('MeanVarianceOptimizer', 'mean_variance', {'risk_aversion': 1.0}),
    ('MaxSharpeOptimizer', 'max_sharpe', {'risk_free_rate': 0.02}),
    ('MinVarianceOptimizer', 'min_variance', {}),
    ('RiskParityOptimizer', 'risk_parity', {}),
    ('BlackLittermanOptimizer', 'black_litterman', {'tau': 0.05, 'confidence': 0.5}),
    ('CVaROptimizer', 'cvar', {'alpha': 0.05, 'n_scenarios': 1000}),
    ('HRPOptimizer', 'hrp', {}),
    ('KellyOptimizer', 'kelly', {'kelly_fraction': 0.5}),
]

results_comparison = []

for old_class_name, new_method_name, kwargs in test_cases:
    print(f"\nTesting: {old_class_name} vs TAAOptimizer(method='{new_method_name}')")
    print("-" * 60)
    
    try:
        # Old version (individual class with max_position parameter)
        old_class = getattr(old_opt, old_class_name)
        old_optimizer = old_class(max_position=0.40, **kwargs)
        old_weights = old_optimizer.optimize(
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            previous_weights=None
        )
        
        old_weights_array = np.array([old_weights[t] for t in tickers])
        
        # New version (factory pattern with constraints)
        new_optimizer = TAAOptimizer(constraints, method=new_method_name, **kwargs)
        new_weights, new_metadata = new_optimizer.optimize(
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            previous_weights=None
        )
        
        new_weights_array = np.array([new_weights[t] for t in tickers])
        
        # Compare results
        weight_diff = np.abs(old_weights_array - new_weights_array)
        max_diff = weight_diff.max()
        mean_diff = weight_diff.mean()
        
        # Check if weights are similar (allow small numerical differences)
        tolerance = 0.01  # 1% tolerance
        passed = max_diff < tolerance
        
        print(f"   Old weights sum: {old_weights_array.sum():.6f}")
        print(f"   New weights sum: {new_weights_array.sum():.6f}")
        print(f"   Max difference: {max_diff:.6f}")
        print(f"   Mean difference: {mean_diff:.6f}")
        print(f"   Status: {'✓ PASS' if passed else '✗ FAIL'}")
        
        results_comparison.append({
            'method': new_method_name,
            'old_class': old_class_name,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'passed': passed,
            'old_sum': old_weights_array.sum(),
            'new_sum': new_weights_array.sum(),
        })
        
        # Show top 3 allocations for comparison
        old_sorted = sorted(old_weights.items(), key=lambda x: x[1], reverse=True)[:3]
        new_sorted = sorted(new_weights.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"\n   Top 3 allocations (old): {[(t, f'{w:.4f}') for t, w in old_sorted]}")
        print(f"   Top 3 allocations (new): {[(t, f'{w:.4f}') for t, w in new_sorted]}")
        
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        results_comparison.append({
            'method': new_method_name,
            'old_class': old_class_name,
            'max_diff': None,
            'mean_diff': None,
            'passed': False,
            'error': str(e)
        })

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

results_df = pd.DataFrame(results_comparison)
print(f"\nTotal tests: {len(results_comparison)}")
print(f"Passed: {results_df['passed'].sum()}")
print(f"Failed: {(~results_df['passed']).sum()}")

if results_df['passed'].all():
    print("\n🎉 ALL TESTS PASSED! Refactored optimizer produces same results.")
    print("   Safe to commit and push.")
    sys.exit(0)
else:
    print("\n⚠️  SOME TESTS FAILED! Review differences before committing.")
    print("\nFailed methods:")
    failed = results_df[~results_df['passed']]
    print(failed[['method', 'old_class', 'max_diff']])
    sys.exit(1)
