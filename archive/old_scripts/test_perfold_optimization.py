"""
Test per-fold parameter optimization in walk-forward validation.
Configure which signal to test in the __main__ section below.
"""
import pandas as pd
from core.backtest_engine import run_walk_forward
from signals.momentum import MomentumSignalV2
from signals.mean_reversion import MeanReversionSignal
from signals.hybrid_adaptive import HybridAdaptiveSignal


def test_perfold_optimization(signal_factory, param_grid, lookback_history):
    """Test per-fold parameter optimization."""
    print("="*70)
    print("TEST: Per-Fold Parameter Optimization")
    print("="*70)
    
    # Load data
    df = pd.read_csv('Dataset/spx_1990_2025.csv', index_col=0, parse_dates=True)
    df = df.sort_index()
    
    print(f"\nDataset: {len(df):,} rows from {df.index[0].date()} to {df.index[-1].date()}")
    
    print("\nParameter Grid:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    
    total_combos = 1
    for values in param_grid.values():
        total_combos *= len(values)
    print(f"\nTotal combinations: {total_combos}")
    
    # Run with per-fold optimization
    print("\n" + "="*70)
    print("Running Walk-Forward with Per-Fold Optimization...")
    print("="*70)
    
    results = run_walk_forward(
        signal_factory=signal_factory,
        df=df,
        train_size=int(len(df) * 0.6),
        test_size=int(len(df) * 0.2),
        lookback=lookback_history,
        initial_cash=100_000,
        transaction_cost=3.0,
        stop_loss_pct=0.10,  # Default (will be overridden per fold)
        save_dir='logs/perfold_optimization_test',
        optimize_per_fold=True,
        param_grid=param_grid,
        optimization_metric='sharpe',
    )
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Overall performance
    print("\nOverall Performance:")
    for key, value in results['overall'].items():
        print(f"  {key:25s}: {value}")
    
    # Per-fold parameters
    if 'fold_params' in results:
        print("\nParameters Selected Per Fold:")
        for fold_num, params in results['fold_params'].items():
            print(f"\n  Fold {fold_num}:")
            for param_name, param_value in params.items():
                print(f"    {param_name:20s}: {param_value}")
    
    # Fold-by-fold performance
    print("\nFold-by-Fold Performance:")
    if results['folds']:
        # Check what keys are available
        first_fold = results['folds'][0]
        print(f"Available keys: {list(first_fold.keys())}\n")
        
        print(f"{'Fold':<6} {'Return %':<10} {'Sharpe':<8} {'Trades':<8}")
        print("-" * 40)
        for fold in results['folds']:
            print(f"{fold['fold']:<6} {fold['fold_return_pct']*100:>8.2f}% {fold['sharpe']:>7.3f} {fold['n_trades']:>7}")
    
    print("\n" + "="*70)
    print("Test Complete!")
    print(f"Reports saved to: logs/perfold_optimization_test/")
    print("="*70)
    
    return results


if __name__ == '__main__':
    # ===================================================================
    # CONFIGURE SIGNAL TO TEST HERE
    # ===================================================================
    
    # Option 1: Momentum Signal with flexible SMA filter (ACTIVE)
    def signal_factory(**kwargs):
        return MomentumSignalV2(
            lookback=kwargs.get('lookback', 100),
            entry_threshold=kwargs.get('entry_threshold', 0.02),
            exit_threshold=kwargs.get('exit_threshold', -0.01),
            sma_filter=kwargs.get('sma_filter', 100),
        )
    
    param_grid = {
        'lookback': [60, 90, 120, 150, 180],        # Momentum lookback: 3-9 months
        'entry_threshold': [0.01, 0.02, 0.03, 0.04, 0.05],  # Entry momentum threshold
        'exit_threshold': [-0.005, -0.01, -0.015, -0.02, -0.025],  # Exit threshold
        'sma_filter': [50, 100, 150],               # Trend filter: 50/100/150 day SMA
        'stop_loss_pct': [0.08, 0.10],              # Risk management
    }
    lookback_history = 200  # Max(180 lookback + 150 SMA) + buffer
    
    # ===================================================================
    
    # Option 2: Mean Reversion Signal
    # def signal_factory(**kwargs):
    #     return MeanReversionSignal(
    #         window=kwargs.get('window', 20),
    #         entry_z=kwargs.get('entry_z', 1.5),
    #         exit_z=kwargs.get('exit_z', 0.5),
    #     )
    # param_grid = {
    #     'window': [20, 30],
    #     'entry_z': [0.7, 0.85, 1.0, 1.25, 2.5],
    #     'exit_z': [0.5, 1.0, 1.5],
    #     'stop_loss_pct': [0.05, 0.06, 0.07, 0.08],
    # }
    # lookback_history = 50
    
    # Option 3: Hybrid Adaptive Signal
    # def signal_factory(**kwargs):
    #     return HybridAdaptiveSignal(
    #         vol_window=50,
    #         vol_threshold=kwargs.get('vol_threshold', 0.010),
    #         mr_window=20,
    #         mr_entry_z=kwargs.get('mr_entry_z', 1.5),
    #         mr_exit_z=0.5,
    #         mom_fast=kwargs.get('mom_fast', 20),
    #         mom_slow=50,
    #     )
    # param_grid = {
    #     'vol_threshold': [0.008, 0.010, 0.012],
    #     'mr_entry_z': [1.0, 1.5, 2.0],
    #     'mom_fast': [10, 20, 30],
    #     'stop_loss_pct': [0.05, 0.08],
    # }
    # lookback_history = 60
    
    # ===================================================================
    
    # Run test
    results = test_perfold_optimization(signal_factory, param_grid, lookback_history)
