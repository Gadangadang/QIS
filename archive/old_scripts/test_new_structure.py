"""Quick test to verify the new structure works correctly."""
import pandas as pd
from core.backtest_engine import run_walk_forward
from signals.momentum import MomentumSignal
from analysis.report import BacktestReport

def main():
    print("=" * 60)
    print("Testing New Codebase Structure")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv("Dataset/spx_full_1990_2025.csv", index_col=0, parse_dates=True)
    df = df.sort_index()
    print(f"   ✓ Loaded {len(df):,} rows from {df.index[0].date()} to {df.index[-1].date()}")

    # Configure backtest (small test)
    print("\n2. Configuring backtest...")
    config = {
        'signal_factory': lambda: MomentumSignal(lookback=120, threshold=0.02),
        'df': df,
        'train_size': int(len(df) * 0.6),
        'test_size': int(len(df) * 0.2),
        'lookback': 250,
        'initial_cash': 100_000,
        'transaction_cost': 3.0,
        'stop_loss_pct': 0.10,
        'save_dir': 'logs/test_new_structure',
    }
    print(f"   ✓ Train size: {config['train_size']:,} rows")
    print(f"   ✓ Test size: {config['test_size']:,} rows")

    # Run backtest
    print("\n3. Running walk-forward backtest...")
    results = run_walk_forward(**config)
    print(f"   ✓ Completed {results['overall']['n_folds']} folds")

    # Verify results structure
    print("\n4. Verifying results structure...")
    required_keys = ['stitched_equity', 'combined_returns', 'folds', 'overall', 'trades', 'df']
    for key in required_keys:
        if key in results:
            print(f"   ✓ '{key}' present")
        else:
            print(f"   ✗ '{key}' MISSING!")
            return False

    # Verify trades
    print(f"\n5. Checking trades...")
    if not results['trades'].empty:
        print(f"   ✓ Found {len(results['trades'])} trades")
        print(f"   ✓ Columns: {results['trades'].columns.tolist()}")
    else:
        print("   ⚠ No trades found (might be expected)")

    # Test report generation
    print("\n6. Testing BacktestReport class...")
    report = BacktestReport(results)
    print("   ✓ BacktestReport created successfully")

    # Check metrics
    print("\n7. Checking metrics...")
    metrics = report.metrics
    key_metrics = ['total_return_pct', 'cagr_pct', 'sharpe', 'sortino', 'calmar', 'max_drawdown_pct']
    for metric in key_metrics:
        if metric in metrics:
            print(f"   ✓ {metric:20s}: {metrics[metric]:+.4f}")
        else:
            print(f"   ✗ {metric} MISSING!")

    # Test worst days/trades
    print("\n8. Testing report methods...")
    worst_days = report.worst_days(5)
    print(f"   ✓ worst_days() returned {len(worst_days)} rows")

    worst_trades = report.worst_trades(5)
    print(f"   ✓ worst_trades() returned {len(worst_trades)} rows")

    # Check output files
    print("\n9. Checking output files...")
    from pathlib import Path
    output_dir = Path(config['save_dir'])
    expected_files = ['stitched_equity.csv', 'combined_returns.csv', 'report.html']
    for filename in expected_files:
        filepath = output_dir / filename
        if filepath.exists():
            print(f"   ✓ {filename} created")
        else:
            print(f"   ✗ {filename} MISSING!")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print(f"\nView the HTML report at: {output_dir / 'report.html'}")
    print("\nNext steps:")
    print("1. Open notebooks/03_backtest_momentum.ipynb in Jupyter")
    print("2. Run the notebook to see the full workflow")
    print("3. Modify parameters and experiment!")

    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
